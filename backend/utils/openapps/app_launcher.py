"""
Application Launcher Module

This module provides functionality to find and launch applications on Windows.
It supports searching in PATH, Program Files, Start Menu, and UWP apps.
"""

import os
import shutil
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# Configure logging
log_dir = Path(__file__).parent.parent.parent / 'logs'  # Points to backend/logs
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger('AppLauncher')
logger.setLevel(logging.INFO)

# Remove all handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add file handler with UTF-8 encoding
file_handler = logging.FileHandler(
    log_dir / 'app_launcher.log',
    encoding='utf-8',
    mode='a'
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

@dataclass
class AppPaths:
    """Container for application path information."""
    start_menu: List[Path]
    program_files: List[Path]
    system_paths: List[str]

class AppLauncher:
    """Handles finding and launching applications on Windows."""
    
    def __init__(self):
        """Initialize with common search paths."""
        self.paths = AppPaths(
            start_menu=[
                Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
                Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs"
            ],
            program_files=[
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
                Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
                Path(os.environ.get("Programs", r"F:\Programs")) / "Programs"
            ],
            system_paths=os.environ.get('PATH', '').split(os.pathsep)
        )

    def log_action(self, app_name: str, message: str, level: str = 'info') -> None:
        """Log application actions with specified log level."""
        log_level = getattr(logger, level.lower(), logger.info)
        log_level(f"{app_name}: {message}")

    def find_in_path(self, app_name: str) -> Optional[Path]:
        """Find application in system PATH."""
        exe_path = shutil.which(app_name)
        return Path(exe_path) if exe_path else None

    def find_in_program_files(self, app_name: str) -> Optional[Path]:
        """Search for application in Program Files directories."""
        app_name = app_name.lower()
        for base in self.paths.program_files:
            try:
                if base.is_dir():
                    for path in base.rglob("*.exe"):
                        if app_name in path.stem.lower():
                            return path
            except (PermissionError, OSError) as e:
                self.log_action(app_name, f"Access error in {base}: {e}", 'warning')
        return None

    def find_in_start_menu(self, app_name: str) -> Optional[Path]:
        """Search for application shortcuts in Start Menu."""
        app_name = app_name.lower()
        for base in self.paths.start_menu:
            try:
                if base.is_dir():
                    for path in base.rglob("*.lnk"):
                        if app_name in path.stem.lower():
                            return path
            except (PermissionError, OSError) as e:
                self.log_action(app_name, f"Access error in Start Menu: {e}", 'warning')
        return None

    def find_uwp_app(self, app_name: str) -> Optional[str]:
        """Find UWP application by name."""
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json"],
                capture_output=True, text=True, check=True, shell=True
            )
            
            import json
            try:
                apps = json.loads(result.stdout)
                if not isinstance(apps, list):
                    apps = [apps]  # Handle case of single result
                
                app_name_lower = app_name.lower()
                for app in apps:
                    if app_name_lower in app.get('Name', '').lower():
                        return app.get('AppID')
            except json.JSONDecodeError:
                # Fallback to string matching if JSON parsing fails
                for line in result.stdout.splitlines():
                    if app_name.lower() in line.lower():
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            return parts[1].strip()
        except subprocess.CalledProcessError as e:
            self.log_action(app_name, f"Failed to query UWP apps: {e}", 'error')
        return None

    def launch_app(self, app_name: str) -> bool:
        """
        Attempt to launch an application using various methods.
        
        Args:
            app_name: Name or partial name of the application to launch
            
        Returns:
            bool: True if the application was launched successfully, False otherwise
        """
        # Try different methods to find and launch the app
        methods = [
            ("PATH", self.find_in_path, None),
            ("Program Files", self.find_in_program_files, None),
            ("Start Menu", self.find_in_start_menu, None),
            ("UWP App", None, self.find_uwp_app)
        ]

        for method_name, path_finder, uwp_finder in methods:
            try:
                if path_finder:
                    path = path_finder(app_name)
                    if path and path.exists():
                        os.startfile(path)
                        self.log_action(app_name, f"✅ Launched via {method_name}: {path}")
                        return True
                elif uwp_finder:
                    app_id = uwp_finder(app_name)
                    if app_id:
                        os.system(f'start shell:AppsFolder\\{app_id}')
                        self.log_action(app_name, f"✅ Launched UWP app: {app_id}")
                        return True
            except Exception as e:
                self.log_action(app_name, f"❌ Error launching via {method_name}: {e}", 'error')
                continue

        self.log_action(app_name, "❌ Application not found", 'warning')
        return False

def launch_application(app_name: str) -> bool:
    """
    Convenience function to launch an application.
    
    Args:
        app_name: Name or partial name of the application to launch
        
    Returns:
        bool: True if the application was launched successfully, False otherwise
    """
    return AppLauncher().launch_app(app_name)

def main():
    """Command-line interface for the application launcher."""
    parser = argparse.ArgumentParser(description='Launch applications on Windows')
    parser.add_argument('app_name', help='Name or partial name of the application to launch')
    args = parser.parse_args()
    
    launcher = AppLauncher()
    success = launcher.launch_app(args.app_name)
    
    if not success:
        print(f"Could not find or launch application: {args.app_name}")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
