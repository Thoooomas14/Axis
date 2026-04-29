import tensorflow_datasets as tfds


def inspect_dataset():
    dataset_name = "fractal20220817_data"
    print(f"Inspecting {dataset_name}...")

    try:
        # Load builder to get info without downloading data (if possible)
        builder = tfds.builder(dataset_name, try_gcs=True)
        print("\n--- Dataset Info ---")
        print(f"Description: {builder.info.description}")
        print(f"Features: {builder.info.features}")
        print(f"Metadata: {builder.info.metadata}")

        # Try to stream one episode to see actual keys
        print("\n--- Streaming One Episode ---")
        ds = tfds.load(
            dataset_name, split="train[:1]", shuffle_files=False, try_gcs=True
        )

        for episode in ds.take(1):
            print("\nEpisode Keys:", episode.keys())

            if "steps" in episode:
                steps = list(episode["steps"])
                print(f"Number of steps: {len(steps)}")
                if len(steps) > 0:
                    first_step = steps[0]
                    print("\nStep Keys:", first_step.keys())

                    if "observation" in first_step:
                        print("\nObservation Keys:", first_step["observation"].keys())
                        for k, v in first_step["observation"].items():
                            print(f"  {k}: {v.shape} {v.dtype}")

                    if "action" in first_step:
                        print("\nAction Keys:", first_step["action"].keys())
                        for k, v in first_step["action"].items():
                            if isinstance(v, dict):
                                print(f"  {k}: (Dict)")
                            else:
                                print(f"  {k}: {v.shape} {v.dtype}")

            # Check for other metadata in episode
            for k, v in episode.items():
                if k != "steps":
                    print(f"\nEpisode Metadata '{k}': {v}")

    except Exception as e:
        print(f"\n❌ Inspection Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_dataset()
