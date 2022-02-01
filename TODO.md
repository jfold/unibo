- sammenlign med random som kun bruges til at plotte x-dist og regret
- regn self.noise_var på stort sæt, plot sqrt(self.noise_var) sammen med regret # done
- scatter-plot: actual impr., expected impr. for hver surrogat plottet for alle epoker,seeds,problemer

  - fjern RS for plot
  - zoom ind fra 0-1 på expected improvement (x-axis)
  - corrcoef

- gem også middelværdi for surrogaten i X[opt_idx]
- Gem runtime for at se hastighed
- Ranking tabel med ingen BO for alle metrikker pånær BO-specifikke
