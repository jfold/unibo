- Akkumuleret varians diagonal af: X^T X således
- Udregn mahanalobis distance mellem x_i og x_bedst
- Rankier hvor explorativt den er
- Ranking tabel med ingen BO for alle metrikker pånær BO-specifikke
- regn self.noise_var på stort sæt, plot sqrt(self.noise_var) sammen med regret # done
- scatter-plot: actual impr., expected impr. for hver surrogat plottet for alle epoker,seeds,problemer

  - fjern RS for plot
  - zoom ind fra 0-1 på expected improvement (x-axis)
  - corrcoef

- gem også middelværdi for surrogaten i X[opt_idx]
- Gem runtime for at se hastighed

Liste af resultater:

SYNTETISKE

- èn samlet figur med alle BO+kalibrerings-metrikker som funktion af epoker
- én rho + p-værdi for ranking mellem kalibreringsfejl og regret
- én rho + p-værdi for numeriske værdier mellem kalibreringsfejl og regret
- én tabel med alle non-bo metrikker metrikker (rækker: surrogat, søjler: metrikker)
- act. improv. vs expected

REAL

- opfølgende: gør GP mere eksplorativt
- akkumuleret regret: hvor god du er og hvor hurtig du er
