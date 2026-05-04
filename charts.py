import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

df = pd.read_csv("resultados.csv")
particulas = df["Partículas"] / 1000

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)

fmt_x = ticker.FuncFormatter(lambda x, _: f"{x:.0f}k")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(
    particulas, df["Tempo CPU (s)"], marker="o", color="#c0392b", label="Sequencial"
)
ax.plot(
    particulas,
    df["Tempo GPU (s)"],
    marker="s",
    color="#1a6fba",
    label="Paralela (CUDA)",
)
ax.set_xlabel("Número de Partículas")
ax.set_ylabel("Tempo de Execução (s)")
ax.xaxis.set_major_formatter(fmt_x)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("imagens/final_tempo.svg", format="svg", bbox_inches="tight")
fig.savefig("imagens/final_tempo.png", format="png", dpi=300, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(particulas, df["Speedup"], marker="D", color="#1e8449", label="Speedup")
ax.axhline(y=20, color="gray", linestyle=":", linewidth=1.2, label="Estabilização ~20×")
ax.axvline(
    x=50,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    alpha=0.6,
    label="Limite de threads (~50k)",
)
ax.set_xlabel("Número de Partículas")
ax.set_ylabel("Speedup (×)")
ax.xaxis.set_major_formatter(fmt_x)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig("imagens/final_speedup.svg", format="svg", bbox_inches="tight")
fig.savefig("imagens/final_speedup.png", format="png", dpi=300, bbox_inches="tight")
plt.close()
