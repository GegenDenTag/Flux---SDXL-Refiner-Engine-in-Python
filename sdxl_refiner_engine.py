import torch
from diffusers import (
    StableDiffusionXLRefinerPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import numpy as np
from typing import Union


class SDXLRefinerEngine:
    """
    Optimierte SDXL Refiner Engine zum Verfeinern von Flux 1 Dev generierten Bildern.
    Entfernt Branding-Effekte und Artefakte durch gezielten Refiner-Durchlauf
    mit niedrigem Denoise-Wert. Unterstützt variable Flux-Auflösungen mit
    Aspect Ratio-Erhaltung (2:3, 9:16, Landscape, etc.).
    """
    
    def __init__(
        self,
        refiner_model_path: str = "./sdxl_v10RefinerVAEFix.safetensors",
        device: str = None,
        torch_dtype=torch.float16,
        num_inference_steps: int = 14,
        guidance_scale: float = 8.0,
        denoise_strength: float = 0.02,
    ):
        """
        Initialisiert die Refiner Engine.
        
        Args:
            refiner_model_path: Pfad zur Refiner-Modell-Datei
            device: Zielgerät ('cuda', 'cpu' oder None für Auto-Erkennung)
            torch_dtype: Torch-Datentyp für die Pipeline
            num_inference_steps: Anzahl Inferenz-Schritte (optimal: 14)
            guidance_scale: Guidance Scale für bessere Qualität (optimal: 8.0)
            denoise_strength: Stärke der Entrauschung (max 0.02 für minimale Änderungen)
        """
        # Geräte-Setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        # Refiner-Parameter
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Denoise-Stärke validieren und begrenzen
        if denoise_strength > 0.02:
            print(f"Warnung: denoise_strength ({denoise_strength}) zu hoch! Wird auf 0.02 begrenzt.")
            self.denoise_strength = 0.02
        else:
            self.denoise_strength = denoise_strength
            
        print(f"Denoise-Stärke: {self.denoise_strength} (niedrig für minimale Bildveränderung)")
        
        # Refiner-Pipeline laden
        self._load_refiner_pipeline(refiner_model_path)
        
    def _load_refiner_pipeline(self, model_path: str):
        """Lädt die Refiner-Pipeline mit optimierten Einstellungen."""
        try:
            self.refiner_pipe = StableDiffusionXLRefinerPipeline.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                from_safetensors=True,
                use_safetensors=True,
            )
            
            # Optimierten Scheduler verwenden für Refiner-Qualität
            # Standard: DPMSolverMultistepScheduler (entspricht ComfyUI dpmpp_2m)
            self.refiner_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.refiner_pipe.scheduler.config,
                use_karras_sigmas=False  # Standard ohne Karras
            )
            
            # Mit Karras-Sigmas für ComfyUI-Äquivalenz (dpmpp_2m + karras):
            # self.refiner_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            #     self.refiner_pipe.scheduler.config,
            #     use_karras_sigmas=True  # Entspricht "karras" scheduler in ComfyUI
            # )
            
            # Alternative für höchste Qualität (langsamer):
            # from diffusers import EulerAncestralDiscreteScheduler
            # self.refiner_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            #     self.refiner_pipe.scheduler.config
            # )
            
            # Alternative für beste Refiner-Performance (empfohlen):
            # from diffusers import DPMSolverSinglestepScheduler  
            # self.refiner_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            #     self.refiner_pipe.scheduler.config
            # )
            
            # Pipeline auf Gerät verschieben
            self.refiner_pipe = self.refiner_pipe.to(self.device)
            
            # Memory-Optimierung für begrenzte VRAM
            if self.device == "cuda":
                self.refiner_pipe.enable_model_cpu_offload()
                self.refiner_pipe.enable_vae_slicing()
                
            print(f"Refiner-Pipeline erfolgreich geladen auf {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der Refiner-Pipeline: {e}")
    
    def _get_compatible_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """
        Konvertiert Flux-Dimensionen zu SDXL-kompatiblen Dimensionen.
        Behält das Aspect Ratio bei und stellt sicher, dass Dimensionen durch 64 teilbar sind.
        
        Args:
            width: Original-Breite
            height: Original-Höhe
            
        Returns:
            Tuple mit SDXL-kompatiblen (width, height)
        """
        # Target Pixel-Count für SDXL
        # Standard: 1M Pixel (1024x1024) - gut für kleinere Flux-Bilder
        # Erhöht: 1.5M Pixel - besser für upscaled Flux-Bilder (1080x1920)
        target_pixels = 1024 * 1024  # Standard: 1M Pixel
        # target_pixels = int(1.5 * 1024 * 1024)  # Optional: 1.5M für größere Bilder
        
        # Aktuelles Aspect Ratio berechnen
        aspect_ratio = width / height
        
        # Neue Dimensionen basierend auf Target-Pixels berechnen
        if aspect_ratio >= 1:  # Landscape oder Square
            new_height = int((target_pixels / aspect_ratio) ** 0.5)
            new_width = int(new_height * aspect_ratio)
        else:  # Portrait (typisch für Flux: 9:16, 2:3)
            new_width = int((target_pixels * aspect_ratio) ** 0.5)
            new_height = int(new_width / aspect_ratio)
        
        # Auf 64 runden (VAE-Requirement für SDXL)
        # Alle Dimensionen müssen durch 64 teilbar sein
        new_width = ((new_width + 31) // 64) * 64
        new_height = ((new_height + 31) // 64) * 64
        
        # Sicherstellen dass wir im SDXL-Bereich bleiben
        # SDXL kann theoretisch bis ~1536px, aber Performance sinkt
        max_dimension = 1536  # SDXL-Empfohlenes Maximum
        if new_width > max_dimension or new_height > max_dimension:
            if new_width > new_height:
                new_width = max_dimension
                new_height = ((new_width / aspect_ratio + 31) // 64) * 64
            else:
                new_height = max_dimension
                new_width = ((new_height * aspect_ratio + 31) // 64) * 64
        
        return int(new_width), int(new_height)
    
    def _image_to_latents(self, image: Image.Image) -> torch.Tensor:
        """
        Konvertiert ein PIL-Bild in Latent-Repräsentation.
        Behält das Aspect Ratio des Original-Flux-Bildes bei.
        
        Args:
            image: PIL-Bild zum Konvertieren
            
        Returns:
            Latent-Tensor für die Refiner-Pipeline
        """
        original_size = image.size
        print(f"Original-Auflösung: {original_size[0]}x{original_size[1]}")
        
        # SDXL-kompatible Dimensionen berechnen
        target_width, target_height = self._get_compatible_dimensions(
            original_size[0], original_size[1]
        )
        print(f"SDXL-Ziel-Auflösung: {target_width}x{target_height}")
        
        # Bild auf SDXL-kompatible Größe skalieren (Aspect Ratio erhalten)
        image_resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Zu Tensor konvertieren und normalisieren
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=self.torch_dtype)
        
        # Normalisierung für VAE [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Durch VAE-Encoder zu Latents
        with torch.no_grad():
            latents = self.refiner_pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.refiner_pipe.vae.config.scaling_factor
            
        return latents
    
    def refine_image(
        self, 
        image: Union[str, Image.Image], 
        prompt: str, 
        seed: int = 42,
        return_original_size: bool = True
    ) -> Image.Image:
        """
        Verfeinert ein Flux 1 Dev generiertes Bild mit dem SDXL-Refiner.
        Verwendet niedrigen Denoise-Wert für minimale Bildveränderungen.
        
        Args:
            image: Eingangsbild (Pfad als String oder PIL.Image)
            prompt: Ursprünglicher Prompt für das Bild
            seed: Seed für reproduzierbare Ergebnisse
            return_original_size: Ob das Bild auf Original-Größe zurückskaliert werden soll
            
        Returns:
            Verfeinertes PIL-Bild (in Original-Auflösung falls return_original_size=True)
        """
        # Bild laden falls Pfad übergeben wurde
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Fehler beim Laden des Bildes: {e}")
        
        # Original-Größe für spätere Rückskalierung merken
        original_size = image.size
        
        # Generator mit Seed erstellen
        generator = torch.Generator(self.device).manual_seed(seed)
        
        try:
            # Bild zu Latents konvertieren (mit Aspect Ratio-Erhaltung)
            print("Konvertiere Flux-Bild zu SDXL-Latent-Repräsentation...")
            latents = self._image_to_latents(image)
            
            # Refiner-Durchlauf mit niedrigem Denoise-Wert
            print(f"Starte Anti-Branding-Verarbeitung (Denoise: {self.denoise_strength})...")
            with torch.autocast(self.device):
                refined_result = self.refiner_pipe(
                    prompt=prompt,
                    image=latents,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    strength=self.denoise_strength,
                    generator=generator,
                )
            
            refined_image = refined_result.images[0]
            
            # Zurück auf Original-Größe skalieren falls gewünscht
            if return_original_size and refined_image.size != original_size:
                print(f"Skaliere zurück auf Original-Größe: {original_size[0]}x{original_size[1]}")
                refined_image = refined_image.resize(original_size, Image.Resampling.LANCZOS)
            
            print("Flux-Branding-Entfernung abgeschlossen!")
            return refined_image
            
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Flux-Bildverfeinerung: {e}")
    
    def cleanup(self):
        """Gibt GPU-Speicher frei."""
        if hasattr(self, 'refiner_pipe'):
            del self.refiner_pipe
            torch.cuda.empty_cache()
            print("GPU-Speicher freigegeben")


# Beispiel-Verwendung für Flux 1 Dev Anti-Branding
if __name__ == "__main__":
    # Engine mit Standardwerten initialisieren (optimal für Flux-Branding-Entfernung)
    refiner = SDXLRefinerEngine(
        refiner_model_path="./sdxl_v10RefinerVAEFix.safetensors"
        # Standardwerte: denoise_strength=0.02, steps=14, guidance=8.0
    )
    
    # Flux 1 Dev Bild verfeinern (typische Flux-Auflösungen)
    flux_image_path = "flux_generated_image.png"  # z.B. 832x1216 (2:3) oder 768x1344 (9:16)
    original_prompt = "A majestic fox in a lush forest, cinematic lighting"
    original_seed = 1234
    
    try:
        refined_image = refiner.refine_image(
            image=flux_image_path,
            prompt=original_prompt, 
            seed=original_seed,
            return_original_size=True  # Behält Flux-Original-Auflösung bei
        )
        
        # Anti-Branding verfeinertes Bild speichern
        refined_image.save("flux_debrand_output.png")
        print("Flux-Anti-Branding Bild gespeichert: flux_debrand_output.png")
        
    except Exception as e:
        print(f"Fehler: {e}")
    
    finally:
        # Speicher freigeben
        refiner.cleanup()
