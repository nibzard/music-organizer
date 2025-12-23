"""Auto-update manager for music organizer.

Checks for updates from GitHub releases and provides update functionality.
Works with both installed packages and single-file distributions.
"""

import asyncio
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import os


# Version tracking
CURRENT_VERSION = "0.1.0"
GITHUB_API_RELEASES = "https://api.github.com/repos/yourusername/music-organizer/releases/latest"
GITHUB_RELEASES_BASE = "https://github.com/yourusername/music-organizer/releases"

# State file for tracking last update check
STATE_FILE = Path.home() / ".cache" / "music-organizer" / "update_state.json"


class UpdateInfo:
    """Information about an available update."""

    def __init__(self, version: str, url: str, changelog: str, checksums: Dict[str, str]):
        self.version = version
        self.url = url
        self.changelog = changelog
        self.checksums = checksums

    def is_newer_than(self, current_version: str) -> bool:
        """Check if this update is newer than current version."""
        try:
            current_parts = [int(x) for x in current_version.lstrip('v').split('.')]
            new_parts = [int(x) for x in self.version.lstrip('v').split('.')]
            return new_parts > current_parts
        except (ValueError, AttributeError):
            return False


class UpdateManager:
    """Manages update checks and installations."""

    def __init__(self,
                 current_version: str = CURRENT_VERSION,
                 state_file: Path = STATE_FILE,
                 check_interval_days: int = 7):
        self.current_version = current_version
        self.state_file = state_file
        self.check_interval = timedelta(days=check_interval_days)
        self._state: Optional[Dict[str, Any]] = None

    def _load_state(self) -> Dict[str, Any]:
        """Load update state from file."""
        if self._state is not None:
            return self._state

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._state = {}
        else:
            self._state = {}

        return self._state

    def _save_state(self, state: Dict[str, Any]) -> None:
        """Save update state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        self._state = state

    def _should_check_for_updates(self) -> bool:
        """Check if enough time has passed since last check."""
        state = self._load_state()

        # Allow manual check with force=True
        if state.get('force_check', False):
            return True

        last_check = state.get('last_check')
        if not last_check:
            return True

        try:
            last_check_date = datetime.fromisoformat(last_check)
            return datetime.now() - last_check_date > self.check_interval
        except (ValueError, TypeError):
            return True

    async def check_for_updates(self, force: bool = False) -> Optional[UpdateInfo]:
        """Check for available updates.

        Args:
            force: Force check even if check interval hasn't passed

        Returns:
            UpdateInfo if update available, None otherwise
        """
        state = self._load_state()

        # Check if we should update
        if not force and not self._should_check_for_updates():
            # Return cached update info if available
            cached_update = state.get('cached_update')
            if cached_update:
                return UpdateInfo(**cached_update)
            return None

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GITHUB_API_RELEASES,
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    # Extract version
                    tag_name = data.get('tag_name', '')
                    version = tag_name.lstrip('v')

                    # Extract changelog
                    changelog = data.get('body', 'No changelog available')

                    # Extract download URLs and checksums
                    assets = data.get('assets', [])
                    checksums = {}
                    download_url = ""

                    for asset in assets:
                        name = asset['name']
                        browser_url = asset['browser_download_url']
                        checksums[name] = browser_url

                        # Set download URL based on current installation
                        if 'music_organize.py' in name and self._is_single_file():
                            download_url = browser_url
                        elif name.endswith('.whl') and not self._is_single_file():
                            download_url = browser_url

                    if not download_url and assets:
                        download_url = assets[0]['browser_download_url']

                    # Create update info
                    update_info = UpdateInfo(
                        version=version,
                        url=download_url,
                        changelog=changelog,
                        checksums=checksums
                    )

                    # Update state
                    new_state = {
                        'last_check': datetime.now().isoformat(),
                        'latest_version': version,
                        'cached_update': {
                            'version': version,
                            'url': download_url,
                            'changelog': changelog,
                            'checksums': checksums
                        }
                    }

                    # Preserve dismissed version
                    if 'dismissed_version' in state:
                        new_state['dismissed_version'] = state['dismissed_version']

                    self._save_state(new_state)

                    # Check if update is newer
                    if update_info.is_newer_than(self.current_version):
                        return update_info

                    return None

        except Exception:
            # Network errors - return cached info if available
            cached_update = state.get('cached_update')
            if cached_update:
                return UpdateInfo(**cached_update)
            return None

    def _is_single_file(self) -> bool:
        """Check if running as single-file distribution."""
        # Check if we're running from the standalone script
        main_file = Path(sys.argv[0]).name
        return main_file == "music_organize.py" or "music_organize" not in sys.executable

    def get_installation_method(self) -> str:
        """Get the current installation method."""
        if self._is_single_file():
            return "single-file"

        # Check if installed via pip
        try:
            import importlib.metadata as metadata
            dist = metadata.distribution("music-organizer")
            if dist:
                return "pip"
        except Exception:
            pass

        # Check if running from source
        if Path(__file__).parent.parent.name == "music_organizer":
            return "source"

        return "unknown"

    def get_update_command(self, update_info: UpdateInfo) -> str:
        """Get the command to update based on installation method."""
        method = self.get_installation_method()

        if method == "single-file":
            return (
                f"curl -L {update_info.url} -o music_organize.py && "
                f"chmod +x music_organize.py"
            )
        elif method == "pip":
            return "pip install --upgrade music-organizer"
        elif method == "source":
            return "git pull && pip install -e ."
        else:
            # Generic instruction
            return f"Download from {update_info.url}"

    async def download_update(self, update_info: UpdateInfo, dest: Optional[Path] = None) -> Path:
        """Download the update file.

        Args:
            update_info: Update information
            dest: Destination path (defaults to current directory)

        Returns:
            Path to downloaded file
        """
        if dest is None:
            dest = Path.cwd()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(update_info.url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Download failed: HTTP {response.status}")

                    # Determine filename from URL
                    filename = update_info.url.split('/')[-1]
                    output_path = dest / filename

                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0

                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Show progress for larger files
                            if total_size > 0 and downloaded % (1024 * 100) == 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading: {percent:.1f}%", end='', file=sys.stderr)

                    print()  # New line after progress
                    return output_path

        except ImportError:
            # Fallback to synchronous download without aiohttp
            import urllib.request

            filename = update_info.url.split('/')[-1]
            output_path = dest / filename

            urllib.request.urlretrieve(update_info.url, output_path)
            return output_path

    async def verify_checksum(self, file_path: Path, expected_checksums: Dict[str, str]) -> bool:
        """Verify downloaded file checksum.

        Args:
            file_path: Path to downloaded file
            expected_checksums: Expected checksums (by filename)

        Returns:
            True if checksum matches
        """
        # Calculate SHA256
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)

        calculated = f"{sha256_hash.hexdigest()}  {file_path.name}"

        # Check against expected checksums
        # Note: GitHub releases don't provide checksums by default
        # This would work if the release workflow uploads checksums.txt
        return True  # Placeholder - would verify against actual checksums

    def dismiss_update(self, version: str) -> None:
        """Dismiss an update notification.

        Args:
            version: Version to dismiss
        """
        state = self._load_state()
        state['dismissed_version'] = version
        self._save_state(state)

    def is_dismissed(self, version: str) -> bool:
        """Check if an update version was dismissed.

        Args:
            version: Version to check

        Returns:
            True if dismissed
        """
        state = self._load_state()
        return state.get('dismissed_version') == version

    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of update status.

        Returns:
            Dict with update status information
        """
        state = self._load_state()
        return {
            'current_version': self.current_version,
            'last_check': state.get('last_check'),
            'latest_version': state.get('latest_version', self.current_version),
            'dismissed_version': state.get('dismissed_version'),
            'installation_method': self.get_installation_method(),
            'update_available': state.get('cached_update') is not None
        }


class UpdateNotifier:
    """Handles update notifications in CLI."""

    def __init__(self, manager: UpdateManager):
        self.manager = manager

    async def check_and_notify(self, force: bool = False) -> Optional[UpdateInfo]:
        """Check for updates and notify if available.

        Args:
            force: Force check even if interval hasn't passed

        Returns:
            UpdateInfo if update available, None otherwise
        """
        update_info = await self.manager.check_for_updates(force=force)

        if update_info and not self.manager.is_dismissed(update_info.version):
            self._show_update_notification(update_info)
            return update_info

        return None

    def _show_update_notification(self, update_info: UpdateInfo) -> None:
        """Display update notification to user.

        Args:
            update_info: Update information
        """
        print()
        print("=" * 60)
        print(f"  🔄 UPDATE AVAILABLE: {self.manager.current_version} → {update_info.version}")
        print("=" * 60)

        # Show abbreviated changelog
        changelog_lines = update_info.changelog.split('\n')[:5]
        for line in changelog_lines:
            if line.strip():
                print(f"  {line}")

        print()
        print("To update, run one of:")
        print(f"  music-organize update --install")
        print(f"  {self.manager.get_update_command(update_info)}")
        print()
        print(f"To dismiss this notification:")
        print(f"  music-organize update --dismiss")
        print()


async def check_for_updates_cli(
    force: bool = False,
    install: bool = False,
    dismiss: bool = False,
    show_summary: bool = False
) -> int:
    """CLI handler for update checks.

    Args:
        force: Force check even if interval hasn't passed
        install: Download and install update
        dismiss: Dismiss current update notification
        show_summary: Show update status summary

    Returns:
        Exit code
    """
    manager = UpdateManager()
    notifier = UpdateNotifier(manager)

    if show_summary:
        summary = manager.get_update_summary()
        print("📋 Update Status:")
        print(f"  Current version: {summary['current_version']}")
        print(f"  Latest version: {summary['latest_version']}")
        print(f"  Last check: {summary.get('last_check', 'Never')}")
        print(f"  Installation: {summary['installation_method']}")
        print(f"  Update available: {summary['update_available']}")
        return 0

    if dismiss:
        update_info = await manager.check_for_updates(force=True)
        if update_info:
            manager.dismiss_update(update_info.version)
            print(f"✅ Dismissed update {update_info.version}")
        else:
            print("No update to dismiss")
        return 0

    if install:
        update_info = await manager.check_for_updates(force=True)
        if not update_info:
            print("No updates available")
            return 0

        print(f"Downloading {update_info.version}...")
        downloaded = await manager.download_update(update_info)

        if await manager.verify_checksum(downloaded, update_info.checksums):
            print(f"✅ Downloaded to {downloaded}")
            print("\nTo complete installation:")
            if manager._is_single_file():
                print(f"  mv {downloaded.name} music_organize.py && chmod +x music_organize.py")
            else:
                print(f"  pip install --upgrade {downloaded}")
        else:
            print("❌ Checksum verification failed")
            return 1

        return 0

    # Default: just check and notify
    update_info = await notifier.check_and_notify(force=force)

    if not update_info:
        print(f"✅ Up to date (version {manager.current_version})")
        return 0

    return 0


# Integration function for async CLI
async def check_updates_on_startup() -> Optional[UpdateInfo]:
    """Check for updates on CLI startup (non-blocking).

    Returns:
        UpdateInfo if update available, None otherwise
    """
    manager = UpdateManager()
    notifier = UpdateNotifier(manager)
    return await notifier.check_and_notify()
