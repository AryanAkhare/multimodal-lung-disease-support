#!/usr/bin/env python3
"""
🚀 MULTIMODAL LUNG DISEASE DETECTION - FULL STACK LAUNCHER
Single command to run the entire application (backend + frontend)

Usage:
    python run_fullstack.py [--backend-only] [--frontend-only] [--skip-install]
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.YELLOW}➜ {text}{Colors.END}")

class FullStackRunner:
    def __init__(self, skip_install=False):
        self.project_root = Path(__file__).parent
        self.skip_install = skip_install
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "web-client"
        self.venv_dir = self.project_root / ".venv"
        self.is_windows = platform.system() == "Windows"
        self.backend_process = None
        self.frontend_process = None
        
    def get_python_executable(self):
        """Get virtual environment Python executable path"""
        if self.is_windows:
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
    
    def run_command(self, cmd, cwd=None, shell=False):
        """Run command and return result"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            print_error(f"Command timed out: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            return False, "", "Timeout"
        except Exception as e:
            print_error(f"Command failed: {str(e)}")
            return False, "", str(e)
    
    def check_requirements(self):
        """Check if Python and Node are installed"""
        print_info("Checking system requirements...")
        
        # Check Python
        success, _, _ = self.run_command([sys.executable, "--version"])
        if not success:
            print_error("Python not found. Please install Python 3.8+")
            return False
        print_success("Python is installed")
        
        # Check Node/npm for frontend
        success, _, _ = self.run_command(["npm", "--version"])
        if not success:
            print_error("npm not found. Please install Node.js: https://nodejs.org/")
            return False
        print_success("npm is installed")
        
        return True
    
    def setup_venv(self):
        """Create or verify virtual environment"""
        if self.venv_dir.exists():
            print_success("Virtual environment already exists")
            return True
        
        print_info("Creating virtual environment...")
        success, out, err = self.run_command([sys.executable, "-m", "venv", str(self.venv_dir)])
        
        if success:
            print_success("Virtual environment created")
            return True
        else:
            print_error(f"Failed to create venv: {err}")
            return False
    
    def install_backend_deps(self):
        """Install backend dependencies"""
        if self.skip_install:
            print_info("Skipping backend dependency installation")
            return True
        
        print_info("Installing backend dependencies...")
        # Use venv Python to install to the virtual environment
        python_exe = str(self.get_python_executable())
        
        success, out, err = self.run_command(
            [python_exe, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"],
            cwd=self.backend_dir
        )
        
        if success:
            print_success("Backend dependencies installed")
            return True
        else:
            print_error(f"Failed to install backend deps: {err}")
            if out:
                print(f"Output: {out}")
            return False
    
    def install_frontend_deps(self):
        """Install frontend dependencies"""
        if self.skip_install:
            print_info("Skipping frontend dependency installation")
            return True
        
        print_info("Installing frontend dependencies...")
        
        success, out, err = self.run_command(
            ["npm", "install"],
            cwd=self.frontend_dir
        )
        
        if success:
            print_success("Frontend dependencies installed")
            return True
        else:
            print_error(f"Failed to install frontend deps: {err}")
            if out:
                print(f"Output: {out}")
            return False
    
    def start_backend(self):
        """Start backend API server"""
        print_info("Starting backend server on http://localhost:8000...")
        
        # Use venv Python executable to ensure dependencies are found
        python_exe = str(self.get_python_executable())
        
        if self.is_windows:
            # Windows: Run in new process with venv Python
            cmd = f'cd /d {self.backend_dir} && "{python_exe}" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000'
            self.backend_process = subprocess.Popen(cmd, shell=True)
        else:
            # Unix/Linux/Mac: Use venv Python
            self.backend_process = subprocess.Popen(
                [python_exe, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                cwd=self.backend_dir
            )
        
        print_success("Backend server started")
        return True
    
    def start_frontend(self):
        """Start frontend dev server"""
        print_info("Starting frontend server on http://localhost:4173...")
        
        if self.is_windows:
            cmd = f'cd /d {self.frontend_dir} && npm run dev'
            self.frontend_process = subprocess.Popen(cmd, shell=True)
        else:
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.frontend_dir
            )
        
        print_success("Frontend server started")
        return True
    
    def wait_for_services(self):
        """Wait for services to be ready"""
        print_info("Waiting for services to start...")
        time.sleep(3)
        
        try:
            import requests
            max_retries = 10
            for i in range(max_retries):
                try:
                    # Check backend
                    resp = requests.get("http://localhost:8000/docs", timeout=2)
                    if resp.status_code == 200:
                        print_success("Backend is ready ✓")
                        break
                except:
                    if i < max_retries - 1:
                        print_info(f"Waiting for backend... ({i+1}/{max_retries})")
                        time.sleep(2)
        except ImportError:
            print_info("Requests library not available, skipping health check")
    
    def monitor_services(self):
        """Monitor running services"""
        print_header("SERVICES RUNNING")
        print(f"{Colors.GREEN}Frontend:{Colors.END} http://localhost:4173")
        print(f"{Colors.GREEN}Backend API:{Colors.END} http://localhost:8000")
        print(f"{Colors.GREEN}API Docs:{Colors.END} http://localhost:8000/docs")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.END}\n")
        
        try:
            if self.backend_process:
                self.backend_process.wait()
            if self.frontend_process:
                self.frontend_process.wait()
        except KeyboardInterrupt:
            print("\n" + Colors.YELLOW + "Shutting down services..." + Colors.END)
            self.stop()
    
    def stop(self):
        """Stop all services"""
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
            except:
                self.backend_process.kill()
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
            except:
                self.frontend_process.kill()
        
        print_success("All services stopped")
    
    def run_backend_only(self):
        """Run only backend"""
        print_header("STARTING BACKEND ONLY")
        
        if not self.check_requirements():
            return False
        
        if not self.install_backend_deps():
            return False
        
        if not self.start_backend():
            return False
        
        self.wait_for_services()
        
        try:
            self.backend_process.wait()
        except KeyboardInterrupt:
            print("\n" + Colors.YELLOW + "Stopping backend..." + Colors.END)
            self.stop()
        
        return True
    
    def run_frontend_only(self):
        """Run only frontend"""
        print_header("STARTING FRONTEND ONLY")
        
        if not self.check_requirements():
            return False
        
        if not self.install_frontend_deps():
            return False
        
        if not self.start_frontend():
            return False
        
        self.wait_for_services()
        
        try:
            self.frontend_process.wait()
        except KeyboardInterrupt:
            print("\n" + Colors.YELLOW + "Stopping frontend..." + Colors.END)
            self.stop()
        
        return True
    
    def run_all(self):
        """Run complete setup and both services"""
        print_header("🚀 MULTIMODAL LUNG DISEASE DETECTION APP")
        print(f"Project: {self.project_root}\n")
        
        # Check system requirements
        if not self.check_requirements():
            return False
        
        # Setup virtual environment
        if not self.setup_venv():
            return False
        
        # Install dependencies
        print_header("INSTALLING DEPENDENCIES")
        if not self.install_backend_deps():
            return False
        
        if not self.install_frontend_deps():
            return False
        
        # Start services
        print_header("STARTING SERVICES")
        
        if not self.start_backend():
            return False
        
        time.sleep(2)  # Give backend time to start
        
        if not self.start_frontend():
            self.stop()
            return False
        
        # Wait and monitor
        self.wait_for_services()
        self.monitor_services()
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 Multimodal Lung Disease Detection - Full Stack Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fullstack.py              # Start everything
  python run_fullstack.py --backend-only    # Start only backend
  python run_fullstack.py --frontend-only   # Start only frontend
  python run_fullstack.py --skip-install    # Don't reinstall dependencies
        """
    )
    
    parser.add_argument("--backend-only", action="store_true", help="Start only backend")
    parser.add_argument("--frontend-only", action="store_true", help="Start only frontend")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    runner = FullStackRunner(skip_install=args.skip_install)
    
    try:
        if args.backend_only:
            success = runner.run_backend_only()
        elif args.frontend_only:
            success = runner.run_frontend_only()
        else:
            success = runner.run_all()
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        runner.stop()
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        runner.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
