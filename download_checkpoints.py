from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

REPO_ID = "nicklashansen/newt"


def list_checkpoints() -> List[str]:
    """
    List available checkpoint filenames in the repo (filters *.pt).
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    return sorted([f for f in files if f.endswith(".pt")])


def download_checkpoint(filename: str, cache_dir: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    Download helper for Newt/TD-MPC2 checkpoints.
    Accepts "atari-pong" or "atari-pong.pt" name conventions.
    """
    if not filename.endswith(".pt") and "/" not in filename:
        # Convenience: support hf_hub_download("atari-pong") name convention
        filename = f"{filename}.pt"

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="config.json",
            revision="main",
            cache_dir=cache_dir,
            token=token,
        )
    except EntryNotFoundError:
        pass

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        revision="main",
        cache_dir=cache_dir,
        token=token,
    )


def download_all_checkpoints(cache_dir: Optional[str] = None, token: Optional[str] = None) -> List[str]:
    """
    Download all available checkpoints in the repo.
    """
    filenames = list_checkpoints()
    paths = []
    for filename in filenames:
        path = download_checkpoint(filename, cache_dir=cache_dir, token=token)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Newt/TD-MPC2 checkpoints from Hugging Face Hub.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available checkpoints.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Specific checkpoint filename to download (e.g., 'atari-pong.pt').",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./checkpoints",
        help="Directory to cache the downloaded checkpoints.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token if needed.",
    )

    args = parser.parse_args()

    if args.all:
        paths = download_all_checkpoints(cache_dir=args.cache_dir, token=args.token)
        print("Downloaded checkpoints:")
        for path in paths:
            print(path)
    elif args.filename:
        path = download_checkpoint(args.filename, cache_dir=args.cache_dir, token=args.token)
        print(f"Downloaded checkpoint: {path}")
    else:
        print("Please specify either --all to download all checkpoints or --filename to download a specific one.")
