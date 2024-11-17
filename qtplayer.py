import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import yt_dlp
import numpy as np
import time
from pygame import mixer
import random
import os

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, video_url, video_window):
        super().__init__()
        self.video_url = video_url
        self.video_window = video_window
        self.running = True
        self.start_time = time.time()
        
        # Extract video title
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                self.video_name = info.get('title', 'Video')
                # Get direct video URL
                for format in info['formats']:
                    if format.get('vcodec', 'none') != 'none':
                        self.video_url = format['url']
                        break
        except Exception as e:
            print(f"Error extracting video info: {e}")
            self.video_name = os.path.basename(video_url)
            self.video_name = self.video_name.replace('-', ' ').replace('_', ' ')
        
        if len(self.video_name) > 30:
            self.video_name = self.video_name[:27] + "..."
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_url)
            if not cap.isOpened():
                self.error.emit("Failed to open video stream")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0/30.0
            
            last_frame_time = time.time()
            last_frame = None
            
            while self.running and cap.isOpened():
                current_time = time.time()
                
                if not self.video_window.is_paused:
                    elapsed = current_time - last_frame_time
                    if elapsed >= frame_delay:
                        ret, frame = cap.read()
                        if ret:
                            last_frame = frame.copy()
                            frame = cv2.resize(frame, (854, 480))
                            elapsed_time = current_time - self.start_time
                            frame = self.add_vcr_overlay(frame, elapsed_time)
                            self.frame_ready.emit(frame)
                            last_frame_time = current_time
                        else:
                            break
                    else:
                        time.sleep(max(0, frame_delay - elapsed))
                else:
                    if last_frame is not None:
                        frame = last_frame.copy()
                        frame = cv2.resize(frame, (854, 480))
                        frame = self.add_vcr_overlay(frame, time.time() - self.video_window.pause_time)
                        self.frame_ready.emit(frame)
                    time.sleep(0.03)
            
            cap.release()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def add_vcr_overlay(self, frame, elapsed_time):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # VCR green color
        VCR_GREEN = (0, 255, 0)  # BGR format
        
        # Only show overlay during first 2 seconds or when paused
        should_show = elapsed_time <= 2.0 or self.video_window.is_paused
        
        if should_show:
            # Play/Pause symbol (top left)
            if self.video_window.is_paused:
                # Draw pause symbol
                cv2.rectangle(overlay, (30, 20), (40, 50), VCR_GREEN, -1)
                cv2.rectangle(overlay, (50, 20), (60, 50), VCR_GREEN, -1)
            else:
                # Draw play triangle
                triangle_pts = np.array([
                    [30, 20],
                    [30, 50],
                    [60, 35]
                ], np.int32)
                cv2.fillPoly(overlay, [triangle_pts], VCR_GREEN)
            
            # Video name (bottom left)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, self.video_name, 
                       (30, height - 30),
                       font, 0.7,
                       VCR_GREEN,
                       2,
                       cv2.LINE_AA)
            
            # Calculate alpha for fading
            alpha = 1.0
            if elapsed_time < 0.5:  # Fade in
                alpha = elapsed_time * 2
            elif elapsed_time > 1.5 and not self.video_window.is_paused:  # Fade out
                alpha = max(0, 2.0 - elapsed_time)
            
            # Apply overlay with alpha
            frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)
            
            # Add green glow
            green_glow = np.zeros_like(frame)
            if self.video_window.is_paused:
                cv2.rectangle(green_glow, (30, 20), (40, 50), (0, 50, 0), -1)
                cv2.rectangle(green_glow, (50, 20), (60, 50), (0, 50, 0), -1)
            else:
                cv2.fillPoly(green_glow, [triangle_pts], (0, 50, 0))
            
            cv2.putText(green_glow, self.video_name, 
                       (30, height - 30),
                       font, 0.7,
                       (0, 50, 0),
                       4,
                       cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1, green_glow, alpha * 0.3, 0)
        
        return frame
    
    def stop(self):
        self.running = False

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video_label = QLabel()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)
        self.setMinimumSize(854, 480)
        
        # Set black background
        self.setStyleSheet("background-color: black;")
        self.video_label.setStyleSheet("background-color: black;")
        
        self.is_paused = False
        self.pause_time = None
        self.is_fullscreen = False
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_time = time.time()
        elif event.key() == Qt.Key.Key_F:
            self.toggle_fullscreen()
        super().keyPressEvent(event)
    
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retro TV Controls")
        self.setFixedSize(400, 250)
        
        # Initialize pygame mixer
        mixer.init()
        
        # Create video window
        self.video_window = VideoWindow()
        self.video_window.show()
        
        # Main widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # URL input
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        layout.addWidget(self.url_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ PLAY")
        self.play_button.clicked.connect(self.start_video)
        self.fullscreen_button = QPushButton("FULLSCREEN")
        self.fullscreen_button.clicked.connect(
            lambda: self.video_window.toggle_fullscreen(None)
        )
        
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.fullscreen_button)
        layout.addLayout(button_layout)
        
        # Effect controls
        effects_layout = QFormLayout()
        
        self.scanline_slider = QSlider(Qt.Orientation.Horizontal)
        self.scanline_slider.setRange(0, 100)
        self.scanline_slider.setValue(30)
        effects_layout.addRow("SCANLINES:", self.scanline_slider)
        
        self.curvature_slider = QSlider(Qt.Orientation.Horizontal)
        self.curvature_slider.setRange(0, 100)
        self.curvature_slider.setValue(15)
        effects_layout.addRow("CURVATURE:", self.curvature_slider)
        
        # Audio controls
        self.vintage_audio = QCheckBox()
        self.vintage_audio.setChecked(True)
        effects_layout.addRow("VINTAGE AUDIO:", self.vintage_audio)
        
        self.static_slider = QSlider(Qt.Orientation.Horizontal)
        self.static_slider.setRange(0, 100)
        self.static_slider.setValue(20)
        effects_layout.addRow("STATIC:", self.static_slider)
        
        layout.addLayout(effects_layout)
        
        # Create a container widget for the status
        status_container = QWidget()
        status_container.setFixedHeight(30)
        status_container.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: none;
            }
        """)
        
        # Status label with proper styling
        self.status_label = QLabel("READY")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00ff33;
                background-color: #000000;
                border: none;
                font-family: 'Menlo';
                font-size: 14px;
                padding: 5px;
            }
        """)
        
        # Add status label to a horizontal layout in the container
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_container)
        
        self.setup_style()
        self.video_thread = None
        
        # Initialize the curvature maps cache
        self.curvature_maps = {}
        self.last_process_time = time.time()
        self.process_interval = 1/30  # 30fps limit for performance

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
            }
            QLabel {
                color: #00ff33;
                font-family: 'Menlo';
                font-size: 14px;
            }
            QPushButton {
                background-color: #111111;
                color: #00ff33;
                border: 2px solid #00ff33;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Menlo';
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #00ff33;
                color: black;
            }
            QLineEdit {
                background-color: #111111;
                color: #00ff33;
                border: 2px solid #00ff33;
                border-radius: 10px;
                padding: 5px;
                font-family: 'Menlo';
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ff33;
                height: 8px;
                background: #111111;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff33;
                border: 1px solid #00ff33;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

    def start_video(self):
        try:
            url = self.url_input.text().strip()
            
            # Fix the URL if user forgot to add https://
            if not url.startswith('http'):
                url = 'https://' + url
            
            self.status_label.setText("LOADING...")
            self.play_button.setEnabled(False)
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'quiet': True,
                'no_warnings': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Get best format with both video and audio
                    formats = info.get('formats', [])
                    best_format = None
                    
                    for f in formats:
                        if f.get('acodec') != 'none' and f.get('vcodec') != 'none':
                            best_format = f
                            break
                    
                    if not best_format:
                        raise Exception("No suitable format found")
                    
                    video_url = best_format['url']
                    
                    # Download audio to temporary file
                    audio_opts = {
                        'format': 'bestaudio',
                        'quiet': True,
                        'outtmpl': 'temp_audio',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                        }]
                    }
                    
                    with yt_dlp.YoutubeDL(audio_opts) as ydl:
                        ydl.download([url])
                    
                    # Start video playback
                    if self.video_thread:
                        self.video_thread.stop()
                        self.video_thread.wait()
                    
                    self.video_thread = VideoThread(video_url, self.video_window)
                    self.video_thread.frame_ready.connect(self.process_frame)
                    self.video_thread.start()
                    
                    # Play audio
                    mixer.music.stop()
                    mixer.music.unload()
                    mixer.music.load('temp_audio.mp3')
                    mixer.music.play()
                    
                    if self.vintage_audio.isChecked():
                        mixer.music.set_volume(0.7)  # Reduced volume for vintage effect
                    
                    self.status_label.setText("PLAYING")
                    
            except Exception as e:
                self.status_label.setText("ERROR: Check URL")
                print(f"Detailed error: {str(e)}")
                
        except Exception as e:
            self.status_label.setText("ERROR: Invalid URL")
            print(f"Detailed error: {str(e)}")
            
        finally:
            self.play_button.setEnabled(True)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        mixer.music.stop()
        mixer.quit()
        
        # Clean up temporary audio file
        import os
        try:
            os.remove('temp_audio.mp3')
        except:
            pass
        
        self.video_window.close()
        event.accept()

    def process_frame(self, frame):
        try:
            # Apply CRT effects
            frame = self.add_crt_effects(frame)
            
            # Convert to RGB for Qt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, 
                            QImage.Format.Format_RGB888)
            
            # Scale to fill window while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_window.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Center the video
            x = (self.video_window.width() - scaled_pixmap.width()) // 2
            y = (self.video_window.height() - scaled_pixmap.height()) // 2
            
            self.video_window.video_label.setGeometry(x, y, 
                                                     scaled_pixmap.width(), 
                                                     scaled_pixmap.height())
            self.video_window.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")

    def add_crt_effects(self, frame):
        # Convert to float32
        frame = frame.astype(np.float32) / 255.0
        
        # SUPER INTENSE SCANLINES
        scanline_intensity = self.scanline_slider.value() / 100.0
        if scanline_intensity > 0:
            scanlines = np.ones_like(frame)
            # Darker scanlines with bright bloom
            scanlines[::2] *= 0.15  # Super dark lines
            scanlines[1::2] *= 1.2   # Overbright lines for that eye-burning effect
            
            # Add vertical scanline drift
            drift = np.sin(time.time() * 2) * 0.5
            scanlines = np.roll(scanlines, int(drift), axis=0)
            
            frame = frame * (1.0 - scanline_intensity + scanline_intensity * scanlines)
        
        # AGGRESSIVE RGB SEPARATION (color bleeding)
        shift_amount = 3  # More noticeable color shift
        frame_r = np.roll(frame[:,:,0], shift_amount, axis=1)
        frame_g = frame[:,:,1]  # Keep green channel centered
        frame_b = np.roll(frame[:,:,2], -shift_amount, axis=1)
        
        # Random horizontal sync jitter
        if random.random() < 0.1:  # 10% chance per frame
            jitter = random.randint(-2, 2)
            frame_r = np.roll(frame_r, jitter, axis=1)
            frame_g = np.roll(frame_g, jitter//2, axis=1)  # Less jitter on green
            frame_b = np.roll(frame_b, jitter-1, axis=1)
        
        frame[:,:,0] = frame_r
        frame[:,:,1] = frame_g
        frame[:,:,2] = frame_b
        
        # PHOSPHOR GLOW AND GHOSTING
        ghost_frame = np.roll(frame, -1, axis=1) * 0.1  # Horizontal ghost
        frame = np.maximum(frame, ghost_frame)  # Blend using maximum for that bloom effect
        
        # Curvature and edge darkening
        curvature = self.curvature_slider.value() / 100.0
        if curvature > 0:
            rows, cols = frame.shape[:2]
            cache_key = f"{rows}_{cols}_{curvature}"
            if not hasattr(self, 'curvature_cache'):
                self.curvature_cache = {}
                
            if cache_key not in self.curvature_cache:
                map_x, map_y = np.meshgrid(np.linspace(0, cols-1, cols),
                                         np.linspace(0, rows-1, rows))
                
                center_x, center_y = cols/2, rows/2
                dx = map_x - center_x
                dy = map_y - center_y
                d = np.sqrt(dx*dx + dy*dy)
                d2 = d*d
                
                # More aggressive edge darkening
                edge_darkness = np.clip(1 - (d2 / (rows*cols)) * 0.7, 0.6, 1.0)
                
                # More pronounced curvature
                f = 1 + d2 * curvature * 0.000005
                
                self.curvature_cache[cache_key] = (
                    (dx*f + center_x).astype(np.float32),
                    (dy*f + center_y).astype(np.float32),
                    edge_darkness.astype(np.float32)
                )
            
            map_x, map_y, edge_darkness = self.curvature_cache[cache_key]
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            frame *= np.dstack([edge_darkness] * 3)
        
        # AUTHENTIC COLOR ADJUSTMENTS
        frame[:,:,2] *= 0.8   # Heavy blue reduction
        frame[:,:,0] = np.clip(frame[:,:,0] * 1.4, 0, 1)  # Aggressive red
        frame[:,:,1] *= 0.9   # Slight green reduction
        
        # Add subtle vertical hold instability
        if random.random() < 0.05:  # 5% chance per frame
            roll_amount = random.randint(-1, 1)
            frame = np.roll(frame, roll_amount, axis=0)
        
        # Contrast boost for that eye-straining look
        frame = np.clip((frame - 0.5) * 1.2 + 0.5, 0, 1)
        
        # Convert back to uint8
        return (frame * 255).astype(np.uint8)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = ControlWindow()
    player.show()
    sys.exit(app.exec())