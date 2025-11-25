import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from typing import Literal

from inspect_ai.dataset import Dataset, Sample

logger = logging.getLogger(__name__)


def build_images(
    samples: Dataset,
    max_workers: int = 4,  # Used for both build_instance_images AND Docker pulls
    force_rebuild: bool = False,
    use_remote_images: bool = True,
    force_arch: Literal["", "arm64", "x86_64"] = "",
    pull_max_workers: int | None = 6,  # Separate control for pull parallelism (defaults to max_workers)
) -> dict[str, str]:
    """This function uses the swe_bench library to build the docker images for the SWE-bench dataset.

    It can also try to pull images from a registry before building them locally.

    Args:
        samples (Dataset): The dataset to build the images for
        max_workers (int): The maximum number of workers to use for building images (passed to build_instance_images). Defaults to 4.
        pull_max_workers (int, optional): Number of parallel workers for Docker pulls. Defaults to max_workers.
            Lower values (2-3) reduce rate limit risk but are slower. Higher values (4-8) are faster but may hit rate limits.
        force_rebuild (bool, optional): Whether to force a rebuild of the images. Defaults to False.
        use_remote_images (bool, optional): Whether to try pulling images from Docker Hub before building locally. Defaults to True. See https://hub.docker.com/u/swebench
        force_arch (str, optional): Optionally force the docker images to be pulled/built for a specific architecture. Defaults to "".
    """
    from docker.client import DockerClient  # type: ignore
    from swebench.harness.docker_build import build_instance_images  # type: ignore

    # NOTE: The changes from swebench 2.1.8 to 3.0.0 are not currently documented, so we use try/except
    # to handle both cases so that we know the code continues to work for 2.x while we establish
    # compatibility with 3.0.x
    try:
        # swebench < 3.0.0
        from swebench.harness.test_spec import make_test_spec  # type: ignore

        extra_build_instance_images_kwargs = {}
    except ImportError:
        # swebench >= 3.0.0
        from swebench.harness.constants import LATEST, SWEbenchInstance  # type: ignore
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

        extra_build_instance_images_kwargs = {"tag": LATEST}

    # IMPORTANT: Clear swe-bench's global logger handlers BEFORE we start logging
    # Otherwise our logs get suppressed
    getLogger().handlers = []  # Swe-bench adds a global logger, which we disable.

    # Ensure our logger is configured to show INFO level messages
    if not logger.handlers:
        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Use print() for critical progress updates that MUST be visible
    # (logger can get suppressed, but print() always shows)
    def log_progress(msg):
        print(f"[PROGRESS] {msg}", flush=True)
        logger.info(msg)  # Also log for structured logging

    build_start = time.time()
    log_progress(f"Starting build_images for {len(samples)} samples (use_remote_images={use_remote_images})")

    # Code copied from the swe_bench repository
    docker_client = DockerClient.from_env()

    # The swebench library requires a huggingface version of the code to be loaded in order to build the images. We load the dataset and then use the library to build the images.
    samples_hf: list[SWEbenchInstance] = [sample_to_hf(s) for s in samples]

    # We also keep a mapping from instance_ids to the name of the docker image
    id_to_docker_image: dict[str, str] = {}
    # Track instances that are invalid (missing required metadata)
    invalid_instance_ids: set[str] = set()

    # Note that remote images are named eg "sphinx-doc_1776_sphinx-11502" instead of "sphinx-doc__sphinx-11502" - they all have "1776" inserted into the ID.
    namespace = "swebench" if use_remote_images else None

    spec_start = time.time()
    for i, swebench_instance in enumerate(samples_hf):
        # Validate required fields before creating test_spec
        if not swebench_instance.get("environment_setup_commit") or not swebench_instance.get("repo"):
            invalid_instance_ids.add(swebench_instance["instance_id"])
            logger.debug(f"Skipping instance {swebench_instance['instance_id']}: missing required metadata")
            continue
        try:
            test_spec = make_test_spec(swebench_instance, namespace=namespace)
            test_spec.arch = force_arch or test_spec.arch
            docker_image_name = test_spec.instance_image_key
            id_to_docker_image[swebench_instance["instance_id"]] = docker_image_name
        except Exception as e:
            invalid_instance_ids.add(swebench_instance["instance_id"])
            logger.warning(f"Failed to create test_spec for instance {swebench_instance['instance_id']}: {e}")
            continue
    spec_time = time.time() - spec_start

    # Get list of locally available Docker images
    available_docker_images = [
        image.tags[0] for image in docker_client.images.list() if len(image.tags) > 0
    ]

    samples_to_build_images_for = [
        s
        for s in samples_hf
        if s["instance_id"] in id_to_docker_image
        and id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]

    log_progress(f"Valid instances: {len(id_to_docker_image)}, Invalid: {len(invalid_instance_ids)}")
    log_progress(f"Images already available: {len(id_to_docker_image) - len(samples_to_build_images_for)}, Need to pull/build: {len(samples_to_build_images_for)}")

    # Try to pull images from Docker Hub first if requested
    if use_remote_images and len(samples_to_build_images_for) > 0:
        # Determine number of parallel workers for pulls
        # Lower = safer for rate limits, higher = faster but riskier
        num_pull_workers = pull_max_workers if pull_max_workers is not None else min(max_workers, 3)
        # Cap at reasonable limit to avoid overwhelming Docker Hub
        num_pull_workers = min(num_pull_workers, 8)

        if num_pull_workers > 1:
            log_progress(f"PULLING {len(samples_to_build_images_for)} images from Docker Hub with {num_pull_workers} parallel workers...")
        else:
            log_progress(f"PULLING {len(samples_to_build_images_for)} images from Docker Hub (sequential, this may take a long time)...")

        pull_start = time.time()
        successfully_pulled = []
        failed_pulls = []

        def pull_single_image(sample_data):
            """Pull a single Docker image. Returns (instance_id, success, error_msg)."""
            i, sample = sample_data
            instance_id = sample["instance_id"]
            image_name = id_to_docker_image[instance_id]

            # Create a new Docker client for this thread (thread-safe)
            thread_docker_client = DockerClient.from_env()

            try:
                pull_single_start = time.time()
                thread_docker_client.images.pull(image_name)
                image_base_name = image_name.split(":")[0]
                thread_docker_client.api.tag(image_name, image_base_name, "latest")
                thread_docker_client.close()
                pull_single_time = time.time() - pull_single_start
                return (instance_id, True, None, pull_single_time)
            except Exception as e:
                thread_docker_client.close()
                error_msg = str(e)
                # Only log warnings for non-404/429 errors to reduce noise
                if "404" not in error_msg and "429" not in error_msg:
                    return (instance_id, False, error_msg, None)
                return (instance_id, False, None, None)  # 404/429 are expected, don't log

        # Use parallel pulls if num_pull_workers > 1, otherwise sequential
        if num_pull_workers > 1:
            # Parallel pulls using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_pull_workers) as executor:
                # Submit all pull tasks
                future_to_sample = {
                    executor.submit(pull_single_image, (i, sample)): (i, sample)
                    for i, sample in enumerate(samples_to_build_images_for)
                }

                # Process results as they complete
                completed = 0
                for future in as_completed(future_to_sample):
                    instance_id, success, error_msg, pull_time = future.result()
                    completed += 1

                    if success:
                        successfully_pulled.append(instance_id)
                        if pull_time:
                            log_progress(f"  ✓ [{completed}/{len(samples_to_build_images_for)}] Pulled {id_to_docker_image[instance_id]} in {pull_time:.1f}s")
                    else:
                        failed_pulls.append(instance_id)
                        if error_msg:
                            logger.warning(f"Failed to pull {id_to_docker_image[instance_id]}: {error_msg}")

                    # Progress update every 10 images
                    if completed % 10 == 0:
                        elapsed = time.time() - pull_start
                        log_progress(f"  Progress: {completed}/{len(samples_to_build_images_for)} images ({elapsed/60:.1f} min elapsed)")
        else:
            # Sequential pulls (original behavior)
            for i, sample in enumerate(samples_to_build_images_for):
                instance_id = sample["instance_id"]
                image_name = id_to_docker_image[instance_id]

                if i > 0 and i % 10 == 0:
                    elapsed = time.time() - pull_start
                    log_progress(f"  Pulled {i}/{len(samples_to_build_images_for)} images... ({elapsed/60:.1f} min elapsed)")

                log_progress(f"  Pulling image {i+1}/{len(samples_to_build_images_for)}: {image_name}")

                try:
                    pull_single_start = time.time()
                    docker_client.images.pull(image_name)
                    image_base_name = image_name.split(":")[0]
                    docker_client.api.tag(image_name, image_base_name, "latest")
                    successfully_pulled.append(instance_id)
                    pull_single_time = time.time() - pull_single_start
                    log_progress(f"  ✓ Successfully pulled {image_name} in {pull_single_time:.1f}s")
                except Exception as e:
                    failed_pulls.append(instance_id)
                    error_msg = str(e)
                    # Only log warnings for non-404/429 errors to reduce noise
                    if "404" not in error_msg and "429" not in error_msg:
                        logger.warning(f"Failed to pull {image_name}: {e}")
                        print(f"[WARNING] Failed to pull {image_name}: {e}", flush=True)

        pull_time = time.time() - pull_start
        log_progress(f"✓ Pulled {len(successfully_pulled)}/{len(samples_to_build_images_for)} images from Docker Hub in {pull_time/60:.1f} min")
        if failed_pulls:
            log_progress(f"  ({len(failed_pulls)} images failed to pull - will be built locally)")

        # Remove successfully pulled images from the build list
        samples_to_build_images_for = [
            s
            for s in samples_to_build_images_for
            if s["instance_id"] not in successfully_pulled
        ]

        # Update available images list
        available_docker_images = [
            image.tags[0]
            for image in docker_client.images.list()
            if len(image.tags) > 0
        ]

    # Build any remaining images locally
    # Filter out invalid instances before building
    valid_samples_to_build = [
        s for s in samples_to_build_images_for
        if s["instance_id"] not in invalid_instance_ids
    ]

    # Double-check: Validate samples the way build_instance_images will process them
    # build_instance_images calls make_test_spec() internally, so we validate here too
    logger.info(f"Validating {len(valid_samples_to_build)} samples before building...")
    final_valid_samples = []
    for sample in valid_samples_to_build:
        is_valid = True
        # Test with no namespace (how build_instance_images likely calls it)
        try:
            test_spec_no_ns = make_test_spec(sample)
            # The assertion happens inside make_test_spec, so if we get here, it passed
            # But double-check the env_image_tag attribute
            if hasattr(test_spec_no_ns, 'env_image_tag') and test_spec_no_ns.env_image_tag is None:
                is_valid = False
                logger.debug(f"Sample {sample['instance_id']} has None env_image_tag")
        except AssertionError as e:
            if "env_image_tag cannot be None" in str(e):
                is_valid = False
                logger.debug(f"Sample {sample['instance_id']} failed assertion: {e}")
            else:
                raise  # Re-raise other assertion errors
        except Exception as e:
            is_valid = False
            logger.debug(f"Sample {sample['instance_id']} failed with exception: {e}")

        if is_valid:
            final_valid_samples.append(sample)
        else:
            invalid_instance_ids.add(sample["instance_id"])
            logger.warning(f"Sample {sample['instance_id']} failed validation for build_instance_images")

    logger.info(f"Validated: {len(final_valid_samples)} valid, {len(invalid_instance_ids)} invalid")

    if invalid_instance_ids:
        logger.warning(f"Skipping {len(invalid_instance_ids)} invalid instances that cannot be built")

    if len(final_valid_samples) > 0:
        logger.info("=" * 80)
        logger.warning(f"BUILDING SWE-BENCH IMAGES for {len(final_valid_samples)} samples")
        logger.warning("  - NOTE: This can take a VERY long time (30+ minutes)")
        logger.info("=" * 80)
        build_local_start = time.time()
        try:
            logger.info(f"Calling build_instance_images with {len(final_valid_samples)} samples, max_workers={max_workers}...")
            build_instance_images(
                client=docker_client,
                dataset=final_valid_samples,  # Only validated samples
                force_rebuild=force_rebuild,
                max_workers=max_workers,
                **extra_build_instance_images_kwargs,
            )
            build_local_time = time.time() - build_local_start
            logger.info(f"✓ Local image building completed in {build_local_time/60:.1f} minutes")
        except AssertionError as e:
            if "env_image_tag cannot be None" in str(e):
                logger.error(f"Still got env_image_tag error despite validation. Processing samples individually to identify the problem...")
                # Process samples one at a time to find the problematic one
                individual_valid_samples = []
                for sample in final_valid_samples:
                    try:
                        build_instance_images(
                            client=docker_client,
                            dataset=[sample],  # Process one at a time
                            force_rebuild=force_rebuild,
                            max_workers=1,
                            **extra_build_instance_images_kwargs,
                        )
                        individual_valid_samples.append(sample)
                    except AssertionError as e2:
                        if "env_image_tag cannot be None" in str(e2):
                            logger.error(f"Sample {sample['instance_id']} causes env_image_tag error, skipping")
                            invalid_instance_ids.add(sample["instance_id"])
                        else:
                            raise

                logger.warning(f"Successfully built images for {len(individual_valid_samples)} out of {len(final_valid_samples)} samples")
            else:
                raise

    # Check that all the images were built (excluding invalid instances)
    available_docker_images = [
        image.tags[0] for image in docker_client.images.list() if len(image.tags) > 0
    ]
    valid_samples_hf = [s for s in samples_hf if s["instance_id"] not in invalid_instance_ids]
    missing_images = [
        id_to_docker_image[s["instance_id"]]
        for s in valid_samples_hf
        if s["instance_id"] in id_to_docker_image and id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]
    bad_instances = [
        s['instance_id']
        for s in valid_samples_hf
        if s["instance_id"] in id_to_docker_image and id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]

    if invalid_instance_ids:
        logger.warning(f"Skipped {len(invalid_instance_ids)} invalid instances: {sorted(list(invalid_instance_ids))[:10]}..." if len(invalid_instance_ids) > 10 else f"Skipped {len(invalid_instance_ids)} invalid instances")

    if bad_instances:
        print(f'{bad_instances=}')
        assert len(missing_images) == 0, (
            f"Not all images were built: {missing_images}"
        )

    total_time = time.time() - build_start
    log_progress(f"✓ build_images completed in {total_time/60:.1f} min ({len(id_to_docker_image)} images mapped)")

    return id_to_docker_image


def sample_to_hf(sample: Sample) -> dict[str, str]:
    assert sample.metadata is not None
    return {
        "problem_statement": str(sample.input),
        "base_commit": sample.metadata["base_commit"],
        "instance_id": str(sample.id),
        "patch": sample.metadata["patch"],
        "PASS_TO_PASS": sample.metadata["PASS_TO_PASS"],
        "FAIL_TO_PASS": sample.metadata["FAIL_TO_PASS"],
        "test_patch": sample.metadata["test_patch"],
        "version": sample.metadata["version"],
        "repo": sample.metadata["repo"],
        "environment_setup_commit": sample.metadata["environment_setup_commit"],
        "hints_text": sample.metadata["hints_text"],
        "created_at": sample.metadata["created_at"],
    }
