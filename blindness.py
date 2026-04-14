import json
import os
import hashlib
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

try:
    import mysql.connector  # optional
except Exception:
    mysql = None

import torch
from model import DEFAULT_LABELS, load_model, predict

USERS_FILE = Path("users.json")

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class UserStore:
    def __init__(self):
        self.mode = "json"
        self.conn = None

        host = os.getenv("DB_HOST")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")

        if host and user and password and db_name and mysql is not None:
            try:
                self.conn = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=db_name,
                )
                self._ensure_table()
                self.mode = "mysql"
            except Exception:
                self.conn = None
                self.mode = "json"

        if self.mode == "json":
            USERS_FILE.touch(exist_ok=True)
            if USERS_FILE.read_text().strip() == "":
                USERS_FILE.write_text(json.dumps({}, indent=2))

    def _ensure_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS thegreat (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE,
                password_hash VARCHAR(255)
            )
            """
        )
        self.conn.commit()

    def create_user(self, username: str, password: str) -> bool:
        if self.mode == "mysql":
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO thegreat (username, password_hash) VALUES (%s, %s)",
                    (username, sha256(password)),
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        data = json.loads(USERS_FILE.read_text())
        if username in data:
            return False
        data[username] = sha256(password)
        USERS_FILE.write_text(json.dumps(data, indent=2))
        return True

    def authenticate(self, username: str, password: str) -> bool:
        if self.mode == "mysql":
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT password_hash FROM thegreat WHERE username=%s", (username,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            return row[0] == sha256(password)
        data = json.loads(USERS_FILE.read_text())
        return data.get(username) == sha256(password)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # UI Setup
        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue")

        self.title("Diabetic Retinova Detection System")
        self.geometry("900x600")
        self.resizable(False, False)
        
        # Deep violet base so frame corners blend with the neon hallway background
       # self.configure(fg_color="#0c0b1c")

        self.user_store = UserStore()
        # Initialize model variables, don't load immediately to speed up UI render
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Background Configuration
        self._set_background()

        self.main_frame = None
        self._build_login()

    def _set_background(self):
        # Read the background image and scale it considering CustomTkinter's dpi scaling
        scale = getattr(self, '_get_window_scaling', lambda: 1.0)()
        self.bg_w = int(900 * scale)
        self.bg_h = int(600 * scale)
        
        self.bg_canvas = ctk.CTkCanvas(self, width=self.bg_w, height=self.bg_h, highlightthickness=0)
        self.bg_canvas.place(relx=0.5, rely=0.5, anchor="center")
        
        bg_paths = [Path("bgs.png"), Path("bg.png"), Path("bg.jpg")]
        bg_path = next((p for p in bg_paths if p.exists()), None)
        
        if bg_path:
            img = Image.open(bg_path).resize((self.bg_w, self.bg_h))
            from PIL import ImageTk
            self.bg_image_tk = ImageTk.PhotoImage(img)
            self.bg_canvas.create_image(self.bg_w // 2, self.bg_h // 2, image=self.bg_image_tk)

    def _build_login(self):
        if self.main_frame:
            self.main_frame.destroy()
        
        if hasattr(self, 'title_id'):
            self.bg_canvas.delete(self.title_id)
            self.bg_canvas.delete(self.subtitle_id)
        
        # Top Header (floating over the background) drawn on canvas for genuine transparency
        # Adjust placement coordinates based on scale
        cx = self.bg_w // 2
        
        # Place headline just beneath the logo printed in the background art
        self.title_id = self.bg_canvas.create_text(
            cx,
            int(self.bg_h * 0.35),
            text="Retinopathy Detection System",
            font=("Arial", 34, "bold"),
            fill="#e6e7ff",
        )
        self.subtitle_id = self.bg_canvas.create_text(
            cx,
            int(self.bg_h * 0.40),
            text="Automated Analysis of Diabetic Eye Images",
            font=("Arial", 18),
            fill="#a7b8ff",
        )
                     
        # Container matching the glassy design
        # Dark glass card with neon border to match the new hallway palette
        self.main_frame = ctk.CTkFrame(
            self,
            width=550,
            height=300,
            fg_color="#14183B",
            border_width=2,
            border_color="#2323ff",
            corner_radius=16,
        )
        self.main_frame.place(relx=0.5, rely=0.68, anchor="center")
        
        self.main_frame.grid_propagate(False)
        self.main_frame.pack_propagate(False)

        # Login form centered (removed legacy eye illustration)
        form_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        form_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            form_frame,
            text="👤Enter your username",
            font=ctk.CTkFont(size=14),
            text_color="#d7ddff",
            fg_color="transparent",
        ).pack(anchor="w", padx=10, pady=(0, 4))
        self.username = ctk.CTkEntry(
            form_frame,
            placeholder_text="username",
            width=320,
            height=45,
            border_width=2,
            border_color="#5cecff",
            corner_radius=10,
            fg_color="#161b36",
            text_color="#f2f3ff",
        )
        self.username.pack(pady=(0, 16))
        
        ctk.CTkLabel(
            form_frame,
            text="🔒Enter your password",
            font=ctk.CTkFont(size=14),
            text_color="#d7ddff",
            fg_color="transparent",
        ).pack(anchor="w", padx=10, pady=(0, 4))
        self.password = ctk.CTkEntry(
            form_frame,
            placeholder_text="",
            width=320,
            height=45,
            show="*",
            border_width=2,
            border_color="#5cecff",
            corner_radius=10,
            fg_color="#161b36",
            text_color="#f2f3ff",
        )
        self.password.pack(pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_frame.pack(pady=4)
        
        ctk.CTkButton(
            btn_frame,
            text="Signup",
            width=140,
            height=45,
            corner_radius=10,
            fg_color="#17255a",
            hover_color="#605acd",
            text_color="#e6e7ff",
            border_width=1,
            border_color="#7f8cff",
            bg_color="#0f1228",
            command=self.signup,
        ).pack(side="left", padx=12)
        ctk.CTkButton(
            btn_frame,
            text="Login",
            width=140,
            height=45,
            corner_radius=10,
            fg_color="#5cecff",
            hover_color="#44c8d6",
            text_color="#0c0f21",
            bg_color="#0f1228",
            command=self.login,
        ).pack(side="left", padx=12)

    def _build_uploader(self):
        if self.main_frame:
            self.main_frame.destroy()
            self.main_frame = None
        
        if hasattr(self, 'title_id'):
            self.bg_canvas.delete(self.title_id)
            self.bg_canvas.delete(self.subtitle_id)

        cx = self.bg_w // 2
        self.title_id_uploader = self.bg_canvas.create_text(
            cx, int(self.bg_h * 0.35), 
            text="Welcome to the System", font=("Arial", 36, "bold"), fill="#e6e7ff"
        )
        self.subtitle_id_uploader = self.bg_canvas.create_text(
            cx, int(self.bg_h * 0.43), 
            text="Upload a retinal image", font=("Arial", 20), fill="#a7b8ff"
        )

        self.upload_btn = ctk.CTkButton(
            self, text="Choose Image to Analyze", width=250, height=50, corner_radius=25,
            fg_color="#2323ff", hover_color="#44c8d6", text_color="#e5e9fb", font=ctk.CTkFont(size=16, weight="bold"),
            command=self.upload_image
        )
        self.upload_btn.place(relx=0.5, rely=0.55, anchor="center")

    def _ensure_model_loaded(self):
        if self.model is None:
            self.model = load_model()

    def login(self):
        user, pw = self.username.get().strip(), self.password.get().strip()
        if not user or not pw:
            messagebox.showwarning("Missing info", "Enter both username and password.")
            return
        if self.user_store.authenticate(user, pw):
            messagebox.showinfo("Success", f"Welcome back, {user}!")
            self._build_uploader()
        else:
            messagebox.showerror("Failed", "Invalid credentials.")

    def signup(self):
        user, pw = self.username.get().strip(), self.password.get().strip()
        if not user or not pw:
            messagebox.showwarning("Missing info", "Enter both username and password.")
            return
        if self.user_store.create_user(user, pw):
            messagebox.showinfo("Account created", "You can now log in.")
        else:
            messagebox.showerror("Failed", "User exists or database error.")

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        try:
            self._ensure_model_loaded()
            label, probs = predict(path, model=self.model, device=self.device, labels=DEFAULT_LABELS)
            conf = max(probs) * 100
            img = Image.open(path).convert("RGB")
            
            # Keep preview reasonable in the dialog
            preview_size = (240, 240)
            img_thumb = img.copy()
            img_thumb.thumbnail(preview_size)
            preview = ctk.CTkImage(img_thumb, size=img_thumb.size)

            # Show stylish result
            result_window = ctk.CTkToplevel(self)
            result_window.title("Analysis Result")
            result_window.geometry("380x260")
            result_window.resizable(False, False)
            result_window.attributes("-topmost", True)
            # Preserve reference to avoid Tk image garbage collection
            result_window.preview_img = preview

            ctk.CTkLabel(result_window, text="Verdict:", font=ctk.CTkFont(size=14)).pack(pady=(20, 5))
            ctk.CTkLabel(result_window, text=label, font=ctk.CTkFont(size=24, weight="bold"), text_color="#b83f3f" if "dr" not in label.lower() else "#259646").pack()
            
            ctk.CTkLabel(result_window, text=f"Confidence: {conf:.1f}%", font=ctk.CTkFont(size=16)).pack(pady=10)
            ctk.CTkButton(
                result_window,
                text="Show plot scan image",
                command=lambda: self._open_scan_tab(preview, label, conf, img.copy()),
                width=180,
            ).pack(pady=(6, 4))
            ctk.CTkButton(result_window, text="OK", command=result_window.destroy, width=100).pack(pady=6)
            
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _open_scan_tab(self, preview_img: ctk.CTkImage, label: str, conf: float, original_img: Image.Image = None):
        """
        Open a new tabbed window dedicated to the scanned image and its prediction.
        """
        scan_window = ctk.CTkToplevel(self)
        scan_window.title("Scan View")
        scan_window.geometry("820x620")
        scan_window.resizable(False, False)
        scan_window.attributes("-topmost", True)
        scan_window.preview_img = preview_img  # keep reference alive

        if original_img is not None:
            try:
                plot_pil = self._generate_plot(original_img)
                # Ensure the plot fits exactly within the view space
                plot_pil.thumbnail((780, 520))
                plot_ctk = ctk.CTkImage(plot_pil, size=plot_pil.size)
                scan_window.plot_ctk = plot_ctk  # keep ref
                ctk.CTkLabel(scan_window, image=plot_ctk, text="").pack(pady=20)
            except Exception as e:
                ctk.CTkLabel(scan_window, text=f"Could not generate plot: {e}").pack(pady=40)
        else:
            ctk.CTkLabel(scan_window, text="Original image not available.").pack(pady=40)

        ctk.CTkButton(scan_window, text="Close", command=scan_window.destroy, width=100).pack(pady=6)

    def _generate_plot(self, original_img_pil):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import numpy as np
        import io

        # Crop and pad logic
        img_np = np.array(original_img_pil)
        if img_np.ndim == 3:
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        elif img_np.ndim == 2:
            gray = img_np
        else:
            gray = img_np

        mask = gray > 7
        if mask.any():
            coords = np.column_stack(np.where(mask))
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0)
            cropped_np = img_np[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
            
            # pad to square
            h, w = cropped_np.shape[:2]
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            if cropped_np.ndim == 3:
                padded = np.pad(cropped_np, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w), (0, 0)), mode='constant')
            else:
                padded = np.pad(cropped_np, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant')
            cropped_pil = Image.fromarray(padded)
        else:
            cropped_pil = original_img_pil.copy()
            
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
        cropped_pil_512 = cropped_pil.resize((512, 512), resample_filter)

        fig = Figure(figsize=(6, 6), dpi=100)
        
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cropped_pil_512)
        ax.set_title('Cropped and Padded Image')
        
        fig.suptitle('Image Processing')
        fig.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
