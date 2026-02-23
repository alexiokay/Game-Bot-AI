"""
DarkOrbit Bot Launcher - Modern GUI with CustomTkinter

No more confusion about which command to run or from which folder!
Just click a button and go.

Right-click any button to configure its arguments.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import threading
import os
import sys
from pathlib import Path
import json

# Ensure we're in the bot directory
BOT_DIR = Path(__file__).parent.absolute()
os.chdir(BOT_DIR)

# Config file for persistent settings
CONFIG_FILE = BOT_DIR / "launcher_config.json"

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"


class ConfigDialog(ctk.CTkToplevel):
    """Modern configuration dialog for command arguments."""

    def __init__(self, parent, title, fields):
        """
        Args:
            parent: Parent window
            title: Dialog title
            fields: List of (name, label, default_value, type) tuples
                   type can be: 'str', 'int', 'bool', 'file', 'folder'
        """
        super().__init__(parent)
        self.result = None
        self.title(title)
        self.geometry("600x500")

        # Center on parent
        self.transient(parent)
        self.grab_set()

        # Make it modal
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        # Main container with scrollable frame
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Scrollable frame for fields
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=550, height=380)
        self.scrollable_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Store fields for reset functionality
        self.fields = fields

        # Create fields
        self.widgets = {}
        for idx, field_data in enumerate(fields):
            # Support both old (name, label, default, type) and new (dict) format
            if isinstance(field_data, dict):
                name = field_data['name']
                label = field_data['label']
                default = field_data['default']
                field_type = field_data['type']
                options = field_data.get('options', [])
                info = field_data.get('info', None)
            else:
                name, label, default, field_type = field_data
                options = []
                info = None

            # Label
            label_widget = ctk.CTkLabel(
                self.scrollable_frame,
                text=label,
                font=ctk.CTkFont(size=13)
            )
            label_widget.grid(row=idx*2, column=0, padx=10, pady=8, sticky="w")

            # Info text (if provided)
            if info:
                info_label = ctk.CTkLabel(
                    self.scrollable_frame,
                    text=f"{info}",
                    font=ctk.CTkFont(size=10),
                    text_color="gray70",
                    anchor="w"
                )
                info_label.grid(row=idx*2+1, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew")

            if field_type == 'bool':
                var = ctk.BooleanVar(value=default)
                widget = ctk.CTkSwitch(
                    self.scrollable_frame,
                    text="",
                    variable=var,
                    width=50
                )
                widget.grid(row=idx*2, column=1, padx=10, pady=8, sticky="w")
                self.widgets[name] = var
            elif field_type == 'choice':
                var = ctk.StringVar(value=default)
                combo = ctk.CTkComboBox(
                    self.scrollable_frame,
                    variable=var,
                    values=options,
                    width=200,
                    state="readonly"
                )
                combo.grid(row=idx*2, column=1, padx=10, pady=8, sticky="w")
                self.widgets[name] = var
            elif field_type == 'file':
                # Container frame for entry + button
                container = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
                container.grid(row=idx*2, column=1, columnspan=2, padx=10, pady=8, sticky="w")

                var = ctk.StringVar(value=default)
                entry = ctk.CTkEntry(
                    container,
                    textvariable=var,
                    width=320
                )
                entry.pack(side="left", padx=(0, 5))

                btn = ctk.CTkButton(
                    container,
                    text="Browse",
                    command=lambda v=var: self._browse_file(v),
                    width=80,
                    height=28
                )
                btn.pack(side="left")
                self.widgets[name] = var
            elif field_type == 'folder':
                # Container frame for entry + button
                container = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
                container.grid(row=idx*2, column=1, columnspan=2, padx=10, pady=8, sticky="w")

                var = ctk.StringVar(value=default)
                entry = ctk.CTkEntry(
                    container,
                    textvariable=var,
                    width=320
                )
                entry.pack(side="left", padx=(0, 5))

                btn = ctk.CTkButton(
                    container,
                    text="Browse",
                    command=lambda v=var: self._browse_folder(v),
                    width=80,
                    height=28
                )
                btn.pack(side="left")
                self.widgets[name] = var
            elif field_type == 'int':
                var = ctk.IntVar(value=default)
                entry = ctk.CTkEntry(
                    self.scrollable_frame,
                    textvariable=var,
                    width=150
                )
                entry.grid(row=idx*2, column=1, padx=10, pady=8, sticky="w")
                self.widgets[name] = var
            else:  # str
                var = ctk.StringVar(value=default)
                entry = ctk.CTkEntry(
                    self.scrollable_frame,
                    textvariable=var,
                    width=350
                )
                entry.grid(row=idx*2, column=1, padx=10, pady=8, sticky="w")
                self.widgets[name] = var

        # Button frame
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        button_frame.grid_columnconfigure(1, weight=1)

        # Reset button (left)
        reset_btn = ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            command=self._reset,
            width=140,
            height=32,
            fg_color="gray40",
            hover_color="gray30"
        )
        reset_btn.grid(row=0, column=0, padx=5, sticky="w")

        # Cancel and OK buttons (right)
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self._cancel,
            width=100,
            height=32,
            fg_color="gray40",
            hover_color="gray30"
        )
        cancel_btn.grid(row=0, column=1, padx=5, sticky="e")

        ok_btn = ctk.CTkButton(
            button_frame,
            text="OK",
            command=self._ok,
            width=100,
            height=32
        )
        ok_btn.grid(row=0, column=2, padx=5, sticky="e")

    def _browse_file(self, var):
        filename = filedialog.askopenfilename(initialdir=BOT_DIR)
        if filename:
            var.set(filename)

    def _browse_folder(self, var):
        folder = filedialog.askdirectory(initialdir=BOT_DIR)
        if folder:
            var.set(folder)

    def _reset(self):
        """Reset all fields to their true hardcoded defaults (not config values)."""
        for field_data in self.fields:
            if isinstance(field_data, dict):
                name = field_data['name']
                true_default = field_data.get('true_default', field_data['default'])
            else:
                name, label, default, field_type = field_data
                true_default = default
            self.widgets[name].set(true_default)

    def _ok(self):
        self.result = {name: widget.get() for name, widget in self.widgets.items()}
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def show(self):
        self.wait_window()
        return self.result


class BotLauncher:
    # Hotkey definitions for each script type
    # These must match exactly what the scripts actually respond to
    HOTKEYS = {
        'bot': [
            ('F1', 'Pause/Resume'),
            ('F2', 'BAD STOP'),
            ('F3', 'EMERGENCY STOP'),
            ('F4', 'Debug'),
            ('F5', 'Reasoning'),
            ('F6', 'Mode'),
        ],
        'shadow': [
            ('F1', 'Pause (disabled)'),
            ('F2', 'BAD STOP'),
            ('F3', 'EMERGENCY STOP'),
            ('F4', 'Debug'),
            ('F5', 'Reasoning'),
            ('F6', 'Mode'),
        ],
        'recording': [
            ('F5', 'Start/Stop'),
            ('F6', 'Save'),
            ('F7', 'Discard'),
        ],
        'training': [],  # No hotkeys during training
        'viewer': [
            ('Q', 'Quit'),
            ('D', 'Debug'),
            ('Space', 'Pause'),
        ],
    }

    def __init__(self, root):
        self.root = root
        self.root.title("DarkOrbit Bot Launcher")
        self.root.geometry("1100x850")

        # Current process
        self.current_process = None
        self.running = False
        self.current_script_type = None  # Track which script is running
        self._debug_viewer_process = None
        self._debug_viewer_launched = False

        # Load config
        self.config = self.load_config()

        # Create UI
        self.create_ui()

        # Handle window close - ALWAYS kill child processes
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Handle window close - kill all child processes before closing."""
        print("[LAUNCHER] Closing - killing all child processes...")
        self._kill_all_processes()
        self.root.destroy()

    def _kill_process_tree(self, proc):
        """Kill a process and all its children (Windows-compatible)."""
        if proc is None:
            return

        try:
            pid = proc.pid
            if pid is None:
                return

            if sys.platform == 'win32':
                # On Windows, use taskkill to kill entire process tree
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True,
                    timeout=5
                )
            else:
                # On Unix, try to kill process group
                import signal
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except:
                    proc.terminate()

            # Wait a bit then force kill if still running
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)

        except Exception as e:
            print(f"[LAUNCHER] Error killing process: {e}")
            # Last resort: try kill()
            try:
                proc.kill()
            except:
                pass

    def _kill_all_processes(self):
        """Kill all tracked child processes."""
        # Kill main process
        if self.current_process:
            print(f"[LAUNCHER] Killing main process (PID: {self.current_process.pid})")
            self._kill_process_tree(self.current_process)
            self.current_process = None

        # Kill debug viewer
        if self._debug_viewer_process:
            print(f"[LAUNCHER] Killing debug viewer (PID: {self._debug_viewer_process.pid})")
            self._kill_process_tree(self._debug_viewer_process)
            self._debug_viewer_process = None

        self.running = False
        self._debug_viewer_launched = False

    def load_config(self):
        """Load saved configuration."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        """Save configuration."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def create_ui(self):
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Title frame
        title_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        title = ctk.CTkLabel(
            title_frame,
            text="DarkOrbit Bot Launcher",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack()

        subtitle = ctk.CTkLabel(
            title_frame,
            text="Right-click any button to configure • Ctrl+F to search output • Ctrl+A to select all",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        subtitle.pack()

        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)

        # Left panel - Buttons (scrollable)
        left_panel = ctk.CTkScrollableFrame(main_container, width=350)
        left_panel.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")

        # === RECORDING SECTION ===
        self._create_section_label(left_panel, "RECORDING")

        self.btn_shadow = ctk.CTkButton(
            left_panel,
            text="Shadow Training (needs models)",
            command=self.run_shadow_training,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#4CAF50",
            hover_color="#45a049",
            text_color="white"
        )
        self.btn_shadow.pack(fill="x", padx=10, pady=5)
        self.btn_shadow.bind("<Button-3>", lambda e: self.config_shadow_training())

        self.btn_record = ctk.CTkButton(
            left_panel,
            text="Pure Recording (no models)",
            command=self.run_pure_recording,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#2196F3",
            hover_color="#1976D2",
            text_color="white"
        )
        self.btn_record.pack(fill="x", padx=10, pady=5)
        self.btn_record.bind("<Button-3>", lambda e: self.config_pure_recording())

        self._create_separator(left_panel)

        # === TRAINING SECTION ===
        self._create_section_label(left_panel, "TRAINING")

        self.btn_train_strategist = ctk.CTkButton(
            left_panel,
            text="Train Strategist",
            command=self.run_train_strategist,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#FF9800",
            hover_color="#F57C00",
            text_color="white"
        )
        self.btn_train_strategist.pack(fill="x", padx=10, pady=5)
        self.btn_train_strategist.bind("<Button-3>", lambda e: self.config_train_strategist())

        self.btn_train_tactician = ctk.CTkButton(
            left_panel,
            text="Train Tactician",
            command=self.run_train_tactician,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#FF9800",
            hover_color="#F57C00",
            text_color="white"
        )
        self.btn_train_tactician.pack(fill="x", padx=10, pady=5)
        self.btn_train_tactician.bind("<Button-3>", lambda e: self.config_train_tactician())

        self.btn_train_executor = ctk.CTkButton(
            left_panel,
            text="Train Executor",
            command=self.run_train_executor,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#FF9800",
            hover_color="#F57C00",
            text_color="white"
        )
        self.btn_train_executor.pack(fill="x", padx=10, pady=5)
        self.btn_train_executor.bind("<Button-3>", lambda e: self.config_train_executor())

        self._create_separator(left_panel)

        # === FINE-TUNING SECTION ===
        self._create_section_label(left_panel, "FINE-TUNING")

        self.btn_finetune_vlm = ctk.CTkButton(
            left_panel,
            text="Fine-tune with VLM",
            command=self.run_finetune_vlm,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#673AB7",
            hover_color="#5E35B1",
            text_color="white"
        )
        self.btn_finetune_vlm.pack(fill="x", padx=10, pady=5)
        self.btn_finetune_vlm.bind("<Button-3>", lambda e: self.config_finetune_vlm())

        self._create_separator(left_panel)

        # === RUN BOT SECTION ===
        self._create_section_label(left_panel, "RUN BOT")

        self.btn_run_bot = ctk.CTkButton(
            left_panel,
            text="Run Bot (Autonomous)",
            command=self.run_bot,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#9C27B0",
            hover_color="#7B1FA2",
            text_color="white"
        )
        self.btn_run_bot.pack(fill="x", padx=10, pady=5)
        self.btn_run_bot.bind("<Button-3>", lambda e: self.config_run_bot())

        self.btn_debug_viewer = ctk.CTkButton(
            left_panel,
            text="Debug Viewer",
            command=self.run_debug_viewer,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#E91E63",
            hover_color="#C2185B",
            text_color="white"
        )
        self.btn_debug_viewer.pack(fill="x", padx=10, pady=5)
        self.btn_debug_viewer.bind("<Button-3>", lambda e: self.config_debug_viewer())

        self._create_separator(left_panel)

        # === ANALYSIS SECTION ===
        self._create_section_label(left_panel, "ANALYSIS")

        self.btn_tensorboard = ctk.CTkButton(
            left_panel,
            text="View TensorBoard",
            command=self.run_tensorboard,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#607D8B",
            hover_color="#546E7A",
            text_color="white"
        )
        self.btn_tensorboard.pack(fill="x", padx=10, pady=5)

        self.btn_data_stats = ctk.CTkButton(
            left_panel,
            text="Data Statistics",
            command=self.show_data_stats,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#607D8B",
            hover_color="#546E7A",
            text_color="white"
        )
        self.btn_data_stats.pack(fill="x", padx=10, pady=5)

        self.btn_evaluate = ctk.CTkButton(
            left_panel,
            text="Evaluate Models",
            command=self.evaluate_models,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#607D8B",
            hover_color="#546E7A",
            text_color="white"
        )
        self.btn_evaluate.pack(fill="x", padx=10, pady=5)

        # STOP button
        self.btn_stop = ctk.CTkButton(
            left_panel,
            text="⬛ STOP",
            command=self.stop_process,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#F44336",
            hover_color="#D32F2F",
            text_color="white",
            state="disabled"
        )
        self.btn_stop.pack(fill="x", padx=10, pady=15)

        # Right panel - Output
        right_panel = ctk.CTkFrame(main_container)
        right_panel.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        output_label = ctk.CTkLabel(
            right_panel,
            text="Output",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        output_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.output = ctk.CTkTextbox(
            right_panel,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a",
            text_color="#00ff00",
            wrap="word"
        )
        self.output.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")

        # Add keyboard shortcuts for output
        self.output.bind("<Control-f>", lambda e: self.search_output())
        self.output.bind("<Control-a>", lambda e: self.select_all_output())

        # Hotkey bar (shows hotkeys for currently running script)
        self.hotkey_frame = ctk.CTkFrame(self.root, height=40, fg_color="gray25")
        self.hotkey_frame.grid(row=2, column=0, sticky="ew")
        self.hotkey_frame.grid_columnconfigure(0, weight=1)

        self.hotkey_label = ctk.CTkLabel(
            self.hotkey_frame,
            text="",
            font=ctk.CTkFont(family="Consolas", size=11),
            anchor="w"
        )
        self.hotkey_label.grid(row=0, column=0, padx=15, pady=8, sticky="ew")

        # Initially hidden
        self.hotkey_frame.grid_remove()

        # Status bar
        status_frame = ctk.CTkFrame(self.root, height=35, fg_color="gray20")
        status_frame.grid(row=3, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, padx=15, pady=5, sticky="w")

    def _show_hotkeys(self, script_type: str):
        """Show hotkey bar for the given script type."""
        self.current_script_type = script_type
        hotkeys = self.HOTKEYS.get(script_type, [])

        if hotkeys:
            # Format hotkeys as "F1: Pause  |  F2: Stop  |  ..."
            hotkey_text = "  |  ".join([f"{key}: {desc}" for key, desc in hotkeys])
            self.hotkey_label.configure(text=f"HOTKEYS:  {hotkey_text}")
            self.hotkey_frame.grid()  # Show the frame
        else:
            self._hide_hotkeys()

    def _hide_hotkeys(self):
        """Hide the hotkey bar."""
        self.current_script_type = None
        self.hotkey_frame.grid_remove()

    def _create_section_label(self, parent, text):
        """Create a section label."""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="gray70"
        )
        label.pack(fill="x", padx=10, pady=(10, 5))

    def _create_separator(self, parent):
        """Create a separator line."""
        sep = ctk.CTkFrame(parent, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=20, pady=10)

    def log(self, message):
        """Add message to output window."""
        self.output.insert("end", message + "\n")
        self.output.see("end")

    def select_all_output(self):
        """Select all text in output."""
        self.output.tag_add("sel", "1.0", "end")
        return "break"  # Prevent default behavior

    def search_output(self):
        """Open search dialog for output."""
        dialog = ctk.CTkInputDialog(
            text="Search in output:",
            title="Search"
        )
        search_term = dialog.get_input()

        if search_term:
            # Remove previous highlights
            self.output.tag_remove("search", "1.0", "end")

            # Search and highlight all occurrences
            start_pos = "1.0"
            count = 0
            while True:
                pos = self.output.search(search_term, start_pos, stopindex="end", nocase=True)
                if not pos:
                    break

                end_pos = f"{pos}+{len(search_term)}c"
                self.output.tag_add("search", pos, end_pos)
                count += 1
                start_pos = end_pos

            # Configure search tag appearance
            self.output.tag_config("search", background="yellow", foreground="black")

            # Show first result
            if count > 0:
                first_match = self.output.search(search_term, "1.0", stopindex="end", nocase=True)
                self.output.see(first_match)
                self.status_label.configure(text=f"Found {count} matches for '{search_term}'")
            else:
                self.status_label.configure(text=f"No matches found for '{search_term}'")

    def run_command(self, cmd, description, script_type: str = None):
        """Run a command in a separate thread.

        Args:
            cmd: Command to run
            description: Description shown in output
            script_type: Type of script for hotkey display ('bot', 'shadow', 'recording', 'training', 'viewer')
        """
        if self.running:
            messagebox.showwarning("Already Running", "A process is already running. Stop it first.")
            return

        self.running = True
        self.status_label.configure(text=f"Running: {description}")
        self.btn_stop.configure(state="normal")
        self.disable_all_buttons()

        # Show hotkeys for this script type
        if script_type:
            self._show_hotkeys(script_type)

        self.output.delete("0.0", "end")
        self.log(f"=== {description} ===")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"Working Directory: {BOT_DIR}")
        self.log("")

        def run():
            try:
                import os
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=0,
                    cwd=str(BOT_DIR),
                    env=env
                )

                # Read output line by line
                for line in self.current_process.stdout:
                    self.root.after(0, self.log, line.rstrip())

                self.current_process.wait()
                exit_code = self.current_process.returncode

                if exit_code == 0:
                    self.root.after(0, self.log, f"\n✓ {description} completed successfully")
                    self.root.after(0, self.status_label.configure, {"text": "Ready"})
                else:
                    self.root.after(0, self.log, f"\n✗ {description} exited with code {exit_code}")
                    self.root.after(0, self.status_label.configure, {"text": f"Failed (exit code {exit_code})"})

            except Exception as e:
                self.root.after(0, self.log, f"\n✗ Error: {e}")
                self.root.after(0, self.status_label.configure, {"text": "Error"})

            finally:
                self.running = False
                self.current_process = None
                self.root.after(0, self.btn_stop.configure, {"state": "disabled"})
                self.root.after(0, self.enable_all_buttons)
                self.root.after(0, self._hide_hotkeys)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def stop_process(self):
        """Stop the current running process and all children."""
        self.log("\n[LAUNCHER] Stopping all processes...")

        # Kill main process tree
        if self.current_process:
            self.log(f"[LAUNCHER] Killing main process (PID: {self.current_process.pid})...")
            self._kill_process_tree(self.current_process)
            self.current_process = None
            self.log("[LAUNCHER] Main process stopped")

        # Kill debug viewer tree
        if self._debug_viewer_process:
            self.log(f"[LAUNCHER] Killing debug viewer (PID: {self._debug_viewer_process.pid})...")
            self._kill_process_tree(self._debug_viewer_process)
            self._debug_viewer_process = None
            self.log("[LAUNCHER] Debug viewer stopped")

        # Reset state
        self.running = False
        self._debug_viewer_launched = False
        self.status_label.configure(text="Stopped")
        self.btn_stop.configure(state="disabled")
        self.enable_all_buttons()
        self._hide_hotkeys()

    def _launch_debug_viewer_subprocess(self, monitor: int):
        """Launch debug viewer as a separate subprocess (once only)."""
        import time

        # Mark that we've already launched it
        if hasattr(self, '_debug_viewer_launched') and self._debug_viewer_launched:
            return
        self._debug_viewer_launched = True

        def launch():
            # Wait for bot to initialize and start broadcasting
            # Bot needs time to: load YOLO model, initialize tracker, start control loop
            self.root.after(0, self.log, "[LAUNCHER] Waiting 3s for bot to initialize before launching debug viewer...")
            time.sleep(3.0)

            cmd = [
                "uv", "run", "python", "-m", "darkorbit_bot.v2.debug_viewer",
                "--ipc",
                "--monitor", str(monitor)
            ]

            try:
                # Launch with visible console so user can see errors
                self._debug_viewer_process = subprocess.Popen(
                    cmd,
                    cwd=str(BOT_DIR),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                self.root.after(0, self.log, "[LAUNCHER] Debug viewer launched - check the new console window")
                self.root.after(0, self.log, "[LAUNCHER] Press Q in viewer window to close it (won't reopen)")
            except Exception as e:
                self.root.after(0, self.log, f"[LAUNCHER] Failed to launch debug viewer: {e}")

        # Launch in thread to not block UI
        threading.Thread(target=launch, daemon=True).start()

    def disable_all_buttons(self):
        """Disable all action buttons except Debug Viewer (can run alongside bot)."""
        for widget in [self.btn_shadow, self.btn_record, self.btn_train_strategist,
                      self.btn_train_tactician, self.btn_train_executor, self.btn_finetune_vlm,
                      self.btn_run_bot, self.btn_tensorboard, self.btn_data_stats, self.btn_evaluate]:
            widget.configure(state="disabled")
        # Debug Viewer stays enabled - it's meant to connect to running bot

    def enable_all_buttons(self):
        """Enable all action buttons."""
        for widget in [self.btn_shadow, self.btn_record, self.btn_train_strategist,
                      self.btn_train_tactician, self.btn_train_executor, self.btn_finetune_vlm,
                      self.btn_run_bot, self.btn_debug_viewer, self.btn_tensorboard, self.btn_data_stats, self.btn_evaluate]:
            widget.configure(state="normal")

    # Configuration dialogs
    def config_shadow_training(self):
        """Configure shadow training options."""
        cfg = self.config.get('shadow_training', {})
        fields = [
            ('policy_dir', 'Policy Directory (optional)', cfg.get('policy_dir', ''), 'folder'),
            ('monitor', 'Monitor Number', cfg.get('monitor', 1), 'int'),
            ('save_recordings', 'Save Recordings', cfg.get('save_recordings', True), 'bool'),
            ('learning_rate', 'Learning Rate', cfg.get('learning_rate', '1e-4'), 'str'),
            ('vlm', 'Enable VLM', cfg.get('vlm', False), 'bool'),
            ('vlm_corrections', 'Save VLM Corrections', cfg.get('vlm_corrections', False), 'bool'),
            {
                'name': 'visual_features',
                'label': 'Visual Features',
                'default': cfg.get('visual_features', True),
                'true_default': True,
                'type': 'bool',
                'info': 'Enable CNN visual encoder for scene understanding + debug heatmap (recommended ON)'
            },
            {
                'name': 'visual_lightweight',
                'label': 'Lightweight Visual Mode',
                'default': cfg.get('visual_lightweight', False),
                'true_default': False,
                'type': 'bool',
                'info': 'Use fast color-based encoder instead of CNN (faster, no GPU needed)'
            },
        ]

        result = ConfigDialog(self.root, "Shadow Training Configuration", fields).show()
        if result:
            self.config['shadow_training'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Shadow training configuration saved!")

    def config_pure_recording(self):
        """Configure pure recording options."""
        cfg = self.config.get('pure_recording', {})
        fields = [
            ('model', 'YOLO Model Path', cfg.get('model', 'F:/dev/bot/best.pt'), 'file'),
            ('monitor', 'Monitor Number', cfg.get('monitor', 1), 'int'),
        ]

        result = ConfigDialog(self.root, "Pure Recording Configuration", fields).show()
        if result:
            self.config['pure_recording'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Pure recording configuration saved!")

    def config_train_strategist(self):
        """Configure strategist training options."""
        cfg = self.config.get('train_strategist', {})
        fields = [
            ('data_dir', 'Data Directory', cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'), 'folder'),
            ('epochs', 'Epochs', cfg.get('epochs', 100), 'int'),
            ('batch_size', 'Batch Size', cfg.get('batch_size', 32), 'int'),
            ('learning_rate', 'Learning Rate', cfg.get('learning_rate', '1e-4'), 'str'),
            ('device', 'Device (cpu or cuda)', cfg.get('device', 'cpu'), 'str'),
        ]

        result = ConfigDialog(self.root, "Strategist Training Configuration", fields).show()
        if result:
            self.config['train_strategist'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Strategist training configuration saved!")

    def config_train_tactician(self):
        """Configure tactician training options."""
        cfg = self.config.get('train_tactician', {})
        fields = [
            ('data_dir', 'Data Directory', cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'), 'folder'),
            ('epochs', 'Epochs', cfg.get('epochs', 80), 'int'),
            ('batch_size', 'Batch Size', cfg.get('batch_size', 64), 'int'),
            ('learning_rate', 'Learning Rate', cfg.get('learning_rate', '1e-4'), 'str'),
            ('device', 'Device (cpu or cuda)', cfg.get('device', 'cuda'), 'str'),
        ]

        result = ConfigDialog(self.root, "Tactician Training Configuration", fields).show()
        if result:
            self.config['train_tactician'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Tactician training configuration saved!")

    def config_train_executor(self):
        """Configure executor training options."""
        cfg = self.config.get('train_executor', {})
        fields = [
            ('data_dir', 'Data Directory', cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'), 'folder'),
            ('epochs', 'Epochs', cfg.get('epochs', 60), 'int'),
            ('batch_size', 'Batch Size', cfg.get('batch_size', 128), 'int'),
            ('learning_rate', 'Learning Rate', cfg.get('learning_rate', '1e-4'), 'str'),
            ('device', 'Device (cpu or cuda)', cfg.get('device', 'cuda'), 'str'),
        ]

        result = ConfigDialog(self.root, "Executor Training Configuration", fields).show()
        if result:
            self.config['train_executor'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Executor training configuration saved!")

    def config_finetune_vlm(self):
        """Configure VLM fine-tuning options."""
        cfg = self.config.get('finetune_vlm', {})

        # Get correction counts for preview (with caching to avoid slowness)
        try:
            import subprocess
            import json

            corrections_path = str(BOT_DIR / cfg.get('corrections_dir', 'darkorbit_bot/data/vlm_corrections_v2'))

            # Cache the counts to avoid re-loading on every dialog open
            if not hasattr(self, '_vlm_correction_cache') or self._vlm_correction_cache.get('path') != corrections_path:
                # Run standalone preview script to avoid heavy imports
                result = subprocess.run(
                    [sys.executable, str(BOT_DIR / 'darkorbit_bot/v2/training/preview_vlm_corrections.py'), corrections_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    counts = json.loads(result.stdout.strip())
                else:
                    counts = {'executor': 0, 'strategist': 0, 'tactician': 0, 'total': 0}
                self._vlm_correction_cache = {'path': corrections_path, 'counts': counts}
            else:
                counts = self._vlm_correction_cache['counts']

            if counts['total'] > 0:
                preview_info = (
                    f"{counts['total']} corrections | "
                    f"Strategist: {counts['strategist']} | "
                    f"Executor: {counts['executor']} | "
                    f"Tactician: {counts['tactician']}"
                )
            else:
                preview_info = "No corrections found in directory"

            # Auto-suggest component
            if counts['strategist'] > counts['executor'] and counts['strategist'] > counts['tactician']:
                recommended = "all"
            elif counts['executor'] > 0 and counts['strategist'] > 0:
                recommended = "all"
            elif counts['executor'] > 0:
                recommended = "executor"
            elif counts['strategist'] > 0:
                recommended = "strategist"
            else:
                recommended = "all"
        except Exception as e:
            print(f"Could not preview corrections: {e}")
            import traceback
            traceback.print_exc()
            preview_info = "Error loading corrections - check console"
            recommended = "all"

        fields = [
            {
                'name': 'corrections_dir',
                'label': 'VLM Corrections Directory',
                'default': cfg.get('corrections_dir', 'darkorbit_bot/data/vlm_corrections_v2'),
                'true_default': 'darkorbit_bot/data/vlm_corrections_v2',
                'type': 'folder',
                'info': preview_info
            },
            {
                'name': 'component',
                'label': 'Component to Fine-tune',
                'default': cfg.get('component', recommended),
                'true_default': 'all',
                'type': 'choice',
                'options': ['all', 'strategist', 'executor', 'tactician'],
                'info': "all = fine-tune all models that have corrections"
            },
            {
                'name': 'epochs',
                'label': 'Training Epochs',
                'default': cfg.get('epochs', 5),
                'true_default': 5,
                'type': 'int'
            },
            {
                'name': 'learning_rate',
                'label': 'Learning Rate',
                'default': cfg.get('learning_rate', '1e-5'),
                'true_default': '1e-5',
                'type': 'str',
                'info': "Lower learning rate for fine-tuning (recommended: 1e-5)"
            },
        ]

        result = ConfigDialog(self.root, "VLM Fine-tuning Configuration", fields).show()
        if result:
            self.config['finetune_vlm'] = result
            self.save_config()
            messagebox.showinfo("Saved", "VLM fine-tuning configuration saved!")

    def config_run_bot(self):
        """Configure bot run options."""
        cfg = self.config.get('run_bot', {})
        fields = [
            ('policy_dir', 'Policy Directory', cfg.get('policy_dir', 'models/v2'), 'folder'),
            ('monitor', 'Monitor Number', cfg.get('monitor', 1), 'int'),
            ('vlm', 'Enable VLM', cfg.get('vlm', False), 'bool'),
            ('vlm_corrections', 'Save VLM Corrections', cfg.get('vlm_corrections', False), 'bool'),
            ('online_learning', 'Enable Online Learning', cfg.get('online_learning', False), 'bool'),
            {
                'name': 'visual_features',
                'label': 'Visual Features',
                'default': cfg.get('visual_features', True),
                'true_default': True,
                'type': 'bool',
                'info': 'Enable CNN visual encoder for scene understanding + debug heatmap (recommended ON)'
            },
            {
                'name': 'visual_lightweight',
                'label': 'Lightweight Visual Mode',
                'default': cfg.get('visual_lightweight', False),
                'true_default': False,
                'type': 'bool',
                'info': 'Use fast color-based encoder instead of CNN (less accurate but faster, no GPU needed)'
            },
            {
                'name': 'dagger',
                'label': 'DAgger (Corrective Learning)',
                'default': cfg.get('dagger', False),
                'true_default': False,
                'type': 'bool',
                'info': 'Learn from human corrections during bot play: when you move mouse/click differently than the bot, those corrections get 3x training weight'
            },
            {
                'name': 'launch_debug_viewer',
                'label': 'Launch Debug Viewer',
                'default': cfg.get('launch_debug_viewer', False),
                'true_default': False,
                'type': 'bool',
                'info': 'Automatically opens debug viewer window when bot starts (shows detections, targets, actions)'
            },
        ]

        result = ConfigDialog(self.root, "Run Bot Configuration", fields).show()
        if result:
            self.config['run_bot'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Run bot configuration saved!")

    def config_debug_viewer(self):
        """Configure debug viewer options."""
        cfg = self.config.get('debug_viewer', {})
        fields = [
            {
                'name': 'mode',
                'label': 'Viewer Mode',
                'default': cfg.get('mode', 'ipc'),
                'true_default': 'ipc',
                'type': 'choice',
                'options': ['ipc', 'standalone'],
                'info': 'IPC: Watch running bot (recommended) | Standalone: Run own bot instance'
            },
            ('monitor', 'Monitor Number', cfg.get('monitor', 1), 'int'),
            ('policy_dir', 'Policy Directory (standalone mode only)', cfg.get('policy_dir', 'models/v2'), 'folder'),
        ]

        result = ConfigDialog(self.root, "Debug Viewer Configuration", fields).show()
        if result:
            self.config['debug_viewer'] = result
            self.save_config()
            messagebox.showinfo("Saved", "Debug viewer configuration saved!")

    # Command runners
    def run_shadow_training(self):
        cfg = self.config.get('shadow_training', {})
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.bot_controller_v2", "--shadow-train"]

        if cfg.get('save_recordings', True):
            cmd.append("--save-recordings")
        if cfg.get('policy_dir'):
            cmd.extend(["--policy-dir", cfg['policy_dir']])
        if cfg.get('monitor'):
            cmd.extend(["--monitor", str(cfg['monitor'])])
        if cfg.get('learning_rate'):
            cmd.extend(["--shadow-lr", cfg['learning_rate']])
        if cfg.get('vlm'):
            cmd.append("--vlm")
        if cfg.get('vlm_corrections'):
            cmd.append("--vlm-corrections")

        # Visual features (default is ON, so only add flag if disabled)
        if not cfg.get('visual_features', True):
            cmd.append("--no-visual")
        elif cfg.get('visual_lightweight', False):
            cmd.append("--visual-lightweight")

        self.run_command(cmd, "Shadow Training + Recording", script_type='shadow')

    def run_pure_recording(self):
        cfg = self.config.get('pure_recording', {})
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.recording.recorder_v2",
               "--model", cfg.get('model', 'F:/dev/bot/best.pt'),
               "--monitor", str(cfg.get('monitor', 1))]
        self.run_command(cmd, "Pure Recording", script_type='recording')

    def run_train_strategist(self):
        cfg = self.config.get('train_strategist', {})
        cmd = [".venv/Scripts/python", "-m", "darkorbit_bot.v2.training.train_strategist",
               "--data", cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'),
               "--epochs", str(cfg.get('epochs', 100)),
               "--batch-size", str(cfg.get('batch_size', 32)),
               "--device", cfg.get('device', 'cuda')]
        if cfg.get('learning_rate'):
            cmd.extend(["--lr", cfg['learning_rate']])
        self.run_command(cmd, "Train Strategist", script_type='training')

    def run_train_tactician(self):
        cfg = self.config.get('train_tactician', {})
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.training.train_tactician",
               "--data", cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'),
               "--epochs", str(cfg.get('epochs', 80)),
               "--batch-size", str(cfg.get('batch_size', 64)),
               "--device", cfg.get('device', 'cuda')]
        if cfg.get('learning_rate'):
            cmd.extend(["--lr", cfg['learning_rate']])
        self.run_command(cmd, "Train Tactician", script_type='training')

    def run_train_executor(self):
        cfg = self.config.get('train_executor', {})
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.training.train_executor",
               "--data", cfg.get('data_dir', 'darkorbit_bot/data/recordings_v2'),
               "--epochs", str(cfg.get('epochs', 60)),
               "--batch-size", str(cfg.get('batch_size', 128)),
               "--device", cfg.get('device', 'cuda')]
        if cfg.get('learning_rate'):
            cmd.extend(["--lr", cfg['learning_rate']])
        self.run_command(cmd, "Train Executor", script_type='training')

    def run_finetune_vlm(self):
        cfg = self.config.get('finetune_vlm', {})
        component = cfg.get('component', 'all')
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.training.finetune_with_vlm",
               "--corrections", cfg.get('corrections_dir', 'data/vlm_corrections_v2'),
               "--component", component,
               "--epochs", str(cfg.get('epochs', 5)),
               "--pretrained-dir", "models/v2",
               "--output-dir", "models/v2"]
        if cfg.get('learning_rate'):
            cmd.extend(["--lr", cfg['learning_rate']])

        desc = "Fine-tune All Models" if component == 'all' else f"Fine-tune {component.title()}"
        self.run_command(cmd, f"{desc} with VLM", script_type='training')

    def run_bot(self):
        cfg = self.config.get('run_bot', {})
        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.bot_controller_v2"]

        if cfg.get('policy_dir'):
            cmd.extend(["--policy-dir", cfg['policy_dir']])
        if cfg.get('monitor'):
            cmd.extend(["--monitor", str(cfg['monitor'])])
        if cfg.get('vlm'):
            cmd.append("--vlm")
        if cfg.get('vlm_corrections'):
            cmd.append("--vlm-corrections")
        if cfg.get('online_learning'):
            cmd.append("--online-learning")

        # Visual features (default is ON, so only add flag if disabled)
        if not cfg.get('visual_features', True):
            cmd.append("--no-visual")
        elif cfg.get('visual_lightweight', False):
            # Lightweight mode only applies if visual is enabled
            cmd.append("--visual-lightweight")
        if cfg.get('dagger'):
            cmd.append("--dagger")

        # Debug broadcasting is now always enabled by default
        # Just check if we should launch the viewer window
        launch_viewer = cfg.get('launch_debug_viewer', False)

        desc = "Run Bot (Autonomous)"
        if launch_viewer:
            desc += " + Debug Viewer"

        self.run_command(cmd, desc, script_type='bot')

        # Launch debug viewer in separate process after bot starts
        if launch_viewer:
            self._launch_debug_viewer_subprocess(cfg.get('monitor', 1))

    def run_debug_viewer(self):
        """Run debug viewer as separate process (can run alongside bot)."""
        # Check if already running
        if self._debug_viewer_process and self._debug_viewer_process.poll() is None:
            self.log("[LAUNCHER] Debug viewer is already running!")
            return

        cfg = self.config.get('debug_viewer', {})
        mode = cfg.get('mode', 'ipc')

        cmd = ["uv", "run", "python", "-m", "darkorbit_bot.v2.debug_viewer"]

        if mode == 'ipc':
            cmd.append("--ipc")
            self.log("\n[LAUNCHER] Launching Debug Viewer (IPC mode)...")
            if not self.running:
                self.log("[INFO] Start the bot first, then viewer will connect automatically")
        else:
            if cfg.get('policy_dir'):
                cmd.extend(["--policy-dir", cfg['policy_dir']])
            self.log("\n[LAUNCHER] Launching Debug Viewer (Standalone mode)...")

        if cfg.get('monitor'):
            cmd.extend(["--monitor", str(cfg['monitor'])])

        try:
            # Launch in separate console window
            self._debug_viewer_process = subprocess.Popen(
                cmd,
                cwd=str(BOT_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self._debug_viewer_launched = True
            self.log(f"[LAUNCHER] Debug viewer launched (PID: {self._debug_viewer_process.pid})")
            self.log("[LAUNCHER] Press Q in viewer window to close it")
        except Exception as e:
            self.log(f"[LAUNCHER] Failed to launch debug viewer: {e}")

    # Analysis functions
    def run_tensorboard(self):
        """Launch TensorBoard to view training metrics."""
        cfg = self.config.get('tensorboard', {})
        logdir = cfg.get('logdir', 'runs')
        port = cfg.get('port', 6006)

        cmd = ["uv", "run", "tensorboard", "--logdir", logdir, "--port", str(port)]

        # Run in background and open browser
        self.log(f"\nStarting TensorBoard on http://localhost:{port}")
        self.log(f"Log directory: {logdir}")
        self.log("Press STOP to close TensorBoard when done.\n")

        self.run_command(cmd, "TensorBoard")

        # Open browser after a short delay
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    def show_data_stats(self):
        """Show statistics about recorded data."""
        data_dir = Path("darkorbit_bot/data/recordings_v2")

        if not data_dir.exists():
            messagebox.showinfo("No Data", "No recordings found in darkorbit_bot/data/recordings_v2")
            return

        # Count files and analyze
        json_files = list(data_dir.glob("**/*.json"))
        json_files = [f for f in json_files if 'sequence_' in f.name or 'recording' in f.name.lower() or 'shadow' in f.name.lower()]

        total_frames = 0
        total_size_mb = 0
        shadow_files = []
        v2_files = []

        self.log("\n=== DATA STATISTICS ===\n")
        self.log(f"Data Directory: {data_dir}\n")

        for file_path in json_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if 'demos' in data:
                    # Shadow Training format
                    frames = len(data['demos'])
                    shadow_files.append((file_path.name, frames, size_mb))
                elif 'states' in data:
                    # V2 Recorder format
                    frames = len(data['states'])
                    v2_files.append((file_path.name, frames, size_mb))
                else:
                    frames = 0

                total_frames += frames
            except Exception as e:
                self.log(f"Warning: Could not read {file_path.name}: {e}")

        self.log(f"Total Files: {len(json_files)}")
        self.log(f"Total Frames: {total_frames:,}")
        self.log(f"Total Size: {total_size_mb:.2f} MB\n")

        if shadow_files:
            self.log("Shadow Training Recordings:")
            for name, frames, size in shadow_files[:10]:  # Show first 10
                self.log(f"  {name}: {frames:,} frames ({size:.2f} MB)")
            if len(shadow_files) > 10:
                self.log(f"  ... and {len(shadow_files) - 10} more")
            self.log("")

        if v2_files:
            self.log("V2 Recorder Recordings:")
            for name, frames, size in v2_files[:10]:  # Show first 10
                self.log(f"  {name}: {frames:,} frames ({size:.2f} MB)")
            if len(v2_files) > 10:
                self.log(f"  ... and {len(v2_files) - 10} more")
            self.log("")

        # Estimate training samples
        if total_frames > 0:
            self.log(f"Estimated Training Samples:")
            self.log(f"  Shadow Training: {sum(f[1] for f in shadow_files):,} samples (1 per demo)")
            self.log(f"  V2 Recorder: ~{total_frames // 16:,} samples (sliding window)\n")

        self.log("======================\n")

    def evaluate_models(self):
        """Evaluate trained models."""
        models_dir = Path("models/v2")

        if not models_dir.exists():
            messagebox.showinfo("No Models", "No models found in models/v2")
            return

        self.log("\n=== MODEL EVALUATION ===\n")

        # Check for each model
        models = {
            'Strategist': models_dir / 'strategist' / 'best_model.pt',
            'Tactician': models_dir / 'tactician' / 'best_model.pt',
            'Executor': models_dir / 'executor' / 'best_model.pt'
        }

        found_models = False
        for model_name, model_path in models.items():
            if model_path.exists():
                found_models = True
                size_mb = model_path.stat().st_size / (1024 * 1024)
                modified = model_path.stat().st_mtime
                from datetime import datetime
                modified_str = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")

                self.log(f"{model_name}:")
                self.log(f"  Path: {model_path}")
                self.log(f"  Size: {size_mb:.2f} MB")
                self.log(f"  Last Modified: {modified_str}")

                # Try to load and show basic info
                try:
                    import torch
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                    model_class = checkpoint.get('model_class', 'Unknown')
                    self.log(f"  Architecture: {model_class}")
                    if model_name == 'Executor':
                        is_v2 = model_class in ('ExecutorV2', 'ExecutorV2WithVisual') or 'mouse_head.0.weight' in checkpoint.get('model_state_dict', {})
                        self.log(f"  V2 (separate heads): {'YES' if is_v2 else 'NO - LEGACY (will auto-migrate)'}")
                    if 'epoch' in checkpoint:
                        self.log(f"  Epoch: {checkpoint['epoch']}")
                    if 'loss' in checkpoint:
                        self.log(f"  Loss: {checkpoint['loss']:.4f}")
                    if 'val_loss' in checkpoint:
                        self.log(f"  Val Loss: {checkpoint['val_loss']:.4f}")
                    if 'accuracy' in checkpoint:
                        self.log(f"  Accuracy: {checkpoint['accuracy']:.2%}")

                except Exception as e:
                    self.log(f"  (Could not load checkpoint details: {e})")

                self.log("")

        if not found_models:
            self.log("No trained models found.")
            self.log("Train models first using the TRAINING buttons.\n")

        self.log("========================\n")


def main():
    root = ctk.CTk()
    app = BotLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
