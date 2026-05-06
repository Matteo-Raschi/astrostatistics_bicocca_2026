import json
import uuid

nb_path = 'Untitled-1.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: weighted linear regression + residuals
cell1_src = [
    "# === Linear Regression su dati mu-z (tecnica da L17) ===\n",
    "# Con errori dmu, usiamo il WEIGHTED least squares\n",
    "# w_i = 1/sigma_i  -->  piu' peso ai punti piu' precisi\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "# Grado 1 --> fit lineare: mu = a*z + b\n",
    "degree = 1\n",
    "coeffs = np.polyfit(z_sample, mu_sample, degree, w=1/dmu)\n",
    "a, b = coeffs\n",
    "print(f'Fit lineare: mu = {a:.3f} * z + {b:.3f}')\n",
    "\n",
    "z_fit  = np.linspace(0, 2, 300)\n",
    "mu_fit = np.polyval(coeffs, z_fit)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "\n",
    "# Left: dati + fit\n",
    "ax = axes[0]\n",
    "ax.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1, label='data')\n",
    "ax.plot(z_fit, mu_fit, 'r-', lw=2,\n",
    "        label=fr'Linear fit$\\,$: $\\mu = {a:.2f}\\,z + {b:.2f}$')\n",
    "ax.set_xlabel('z')\n",
    "ax.set_ylabel(r'$\\mu$')\n",
    "ax.set_xlim(0, 2)\n",
    "ax.set_ylim(35, 50)\n",
    "ax.legend(loc='lower right')\n",
    "ax.set_title('Linear Regression (weighted, deg=1)')\n",
    "\n",
    "# Right: residui\n",
    "ax = axes[1]\n",
    "mu_pred   = np.polyval(coeffs, z_sample)\n",
    "residuals = mu_sample - mu_pred\n",
    "ax.errorbar(z_sample, residuals, dmu, fmt='.k', ecolor='gray', lw=1)\n",
    "ax.axhline(0, color='r', lw=2, ls='--', label='zero')\n",
    "ax.set_xlabel('z')\n",
    "ax.set_ylabel(r'$\\mu - \\hat{\\mu}$')\n",
    "ax.set_title('Residuals (linear fit)')\n",
    "ax.set_xlim(0, 2)\n",
    "ax.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

# Cell 2: compare polynomial degrees
cell2_src = [
    "# === Confronto gradi polinomiali (bias-variance tradeoff, da L17) ===\n",
    "\n",
    "degrees = [1, 3, 5]\n",
    "z_fit   = np.linspace(0, 2, 300)\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "for ax, d in zip(axes, degrees):\n",
    "    coeffs  = np.polyfit(z_sample, mu_sample, d, w=1/dmu)\n",
    "    mu_fit  = np.polyval(coeffs, z_fit)\n",
    "    mu_pred = np.polyval(coeffs, z_sample)\n",
    "    # RMS pesato (chi-like)\n",
    "    rms = np.sqrt(np.mean(((mu_sample - mu_pred)/dmu)**2))\n",
    "\n",
    "    ax.errorbar(z_sample, mu_sample, dmu,\n",
    "                fmt='.k', ecolor='gray', lw=1, label='data')\n",
    "    ax.plot(z_fit, mu_fit, 'r-', lw=2, label=f'deg {d}')\n",
    "    ax.set_xlabel('z')\n",
    "    ax.set_ylabel(r'$\\mu$')\n",
    "    ax.set_xlim(0, 2)\n",
    "    ax.set_ylim(35, 50)\n",
    "    ax.legend(loc='lower right')\n",
    "    ax.set_title(f'Grado {d}  |  weighted RMS = {rms:.2f}')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

new_cells = [
    {
        'cell_type': 'code',
        'execution_count': None,
        'id': str(uuid.uuid4())[:8],
        'metadata': {},
        'outputs': [],
        'source': cell1_src,
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'id': str(uuid.uuid4())[:8],
        'metadata': {},
        'outputs': [],
        'source': cell2_src,
    },
]

nb['cells'].extend(new_cells)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Done - added {len(new_cells)} cells. Total cells: {len(nb["cells"])}')
