Optimierte SDXL Refiner Engine zum Verfeinern von Flux.1 Dev generierten Bildern.
    Entfernt Branding-Effekte und Artefakte durch gezielten Refiner-Durchlauf
    mit niedrigem Denoise-Wert. Unterstützt variable Flux-Auflösungen mit
    Aspect Ratio-Erhaltung (2:3, 9:16, Landscape, etc.).

ComfyUI Custom Node

<img width="441" height="464" alt="Screenshot 2025-09-21 032944" src="https://github.com/user-attachments/assets/2aacf31d-45d0-4fd0-897d-be5e02bd902d" />
<br/>
<table>
  <thead>
    <tr>
      <th><strong>Diffusers (Script)</strong></th>
      <th><strong>ComfyUI KSampler</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>DPMSolverMultistepScheduler</code></td>
      <td><code>sampler: dpmpp_2m</code></td>
    </tr>
    <tr>
      <td><code>use_karras_sigmas=False</code></td>
      <td><code>scheduler: normal/simple</code></td>
    </tr>
    <tr>
      <td><code>use_karras_sigmas=True</code></td>
      <td><code>scheduler: karras</code></td>
    </tr>
  </tbody>
</table>
