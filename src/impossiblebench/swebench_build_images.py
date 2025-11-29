import logging
from logging import getLogger
from typing import Literal

from inspect_ai.dataset import Dataset, Sample

logger = logging.getLogger(__name__)


def build_images(
    samples: Dataset,
    max_workers: int = 4,
    force_rebuild: bool = False,
    use_remote_images: bool = True,
    force_arch: Literal["", "arm64", "x86_64"] = "",
) -> dict[str, str]:
    """This function uses the swe_bench library to build the docker images for the SWE-bench dataset.

    It can also try to pull images from a registry before building them locally.

    Args:
        samples (Dataset): The dataset to build the images for
        max_workers (int): The maximum number of workers to use for building images. Defaults to 4.
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

    # Configure logging to show progress
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    # Note: We keep swebench's logger disabled to avoid spam, but use our own logger
    getLogger().handlers = []  # Swe-bench adds a global logger, which we disable.
    # Code copied from the swe_bench repository
    docker_client = DockerClient.from_env()

    # The swebench library requires a huggingface version of the code to be loaded in order to build the images. We load the dataset and then use the library to build the images.
    samples_hf: list[SWEbenchInstance] = [sample_to_hf(s) for s in samples]

    # We also keep a mapping from instance_ids to the name of the docker image
    id_to_docker_image: dict[str, str] = {}

    # Note that remote images are named eg "sphinx-doc_1776_sphinx-11502" instead of "sphinx-doc__sphinx-11502" - they all have "1776" inserted into the ID.
    namespace = "swebench" if use_remote_images else None

    for swebench_instance in samples_hf:
        test_spec = make_test_spec(swebench_instance, namespace=namespace)
        test_spec.arch = force_arch or test_spec.arch
        docker_image_name = test_spec.instance_image_key
        id_to_docker_image[swebench_instance["instance_id"]] = docker_image_name

    # Get list of locally available Docker images
    print(f"\n{'='*60}")
    print(f"Checking Docker images for {len(samples_hf)} samples...")
    print(f"{'='*60}")
    available_docker_images = [
        image.tags[0] for image in docker_client.images.list() if len(image.tags) > 0
    ]
    samples_to_build_images_for = [
        s
        for s in samples_hf
        if id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]
    print(f"Images already available: {len(samples_hf) - len(samples_to_build_images_for)}/{len(samples_hf)}")
    print(f"Images to build/pull: {len(samples_to_build_images_for)}/{len(samples_hf)}")
    print(f"{'='*60}\n")

    # Try to pull images from Docker Hub first if requested
    if use_remote_images and len(samples_to_build_images_for) > 0:
        print(f"\n{'='*60}")
        print(f"Attempting to pull {len(samples_to_build_images_for)} images from Docker Hub...")
        print(f"{'='*60}")
        successfully_pulled = []
        failed_pulls = []

        for i, sample in enumerate(samples_to_build_images_for, 1):
            instance_id = sample["instance_id"]
            image_name = id_to_docker_image[instance_id]
            # Extract just the image name without the tag
            image_base_name = image_name.split(":")[0]

            print(f"[{i}/{len(samples_to_build_images_for)}] Pulling {image_name}...", end=" ", flush=True)
            try:
                docker_client.images.pull(image_name)
                # Tag the pulled image with the expected name
                docker_client.api.tag(image_name, image_base_name, "latest")
                successfully_pulled.append(instance_id)
                print("✓ Success")
            except Exception as e:
                failed_pulls.append((image_name, str(e)))
                print(f"✗ Failed: {e}")

        print(f"\nPulled {len(successfully_pulled)}/{len(samples_to_build_images_for)} images from Docker Hub")
        if failed_pulls:
            print(f"Failed to pull {len(failed_pulls)} images - will build locally")
        print(f"{'='*60}\n")

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
    if len(samples_to_build_images_for) > 0:
        print(f"\n{'='*60}")
        print(f"BUILDING {len(samples_to_build_images_for)} SWE-BENCH IMAGES LOCALLY")
        print(f"NOTE: This can take a VERY long time (hours for 100+ images)")
        print(f"Using {max_workers} parallel workers")
        print(f"{'='*60}\n")

        # Update the list to exclude successfully pulled images
        samples_to_build_images_for = [
            s
            for s in samples_to_build_images_for
            if s["instance_id"] not in successfully_pulled
        ]

        if len(samples_to_build_images_for) > 0:
            print(f"Building {len(samples_to_build_images_for)} images...")
            build_instance_images(
                client=docker_client,
                dataset=samples_hf,
                force_rebuild=force_rebuild,
                max_workers=max_workers,
                **extra_build_instance_images_kwargs,
            )
            print(f"\nBuild complete!")
        else:
            print("All images were successfully pulled - no local builds needed.")

    # Check that all the images were built
    print(f"\n{'='*60}")
    print("Verifying all images are available...")
    print(f"{'='*60}")
    available_docker_images = [
        image.tags[0] for image in docker_client.images.list() if len(image.tags) > 0
    ]
    missing_images = [
        id_to_docker_image[s["instance_id"]]
        for s in samples_hf
        if id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]
    bad_instances = [
        s['instance_id']
        for s in samples_hf
        if id_to_docker_image[s["instance_id"]] not in available_docker_images
    ]

    if bad_instances:
        print(f"WARNING: {len(bad_instances)} images are missing: {bad_instances[:5]}...")
    else:
        print(f"✓ All {len(samples_hf)} images are available!")
    print(f"{'='*60}\n")

    assert len(missing_images) == 0, (
        f"Not all images were built: {missing_images[:10]}..." +
        (f" (and {len(missing_images) - 10} more)" if len(missing_images) > 10 else "")
    )

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
