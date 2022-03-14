Liste af resultater:

SYNTETISKE

- ✓ èn samlet figur med alle BO+kalibrerings-metrikker som funktion af epoker
- ✓ én rho + p-værdi for ranking mellem kalibreringsfejl og regret
- ✓ én rho + p-værdi for numeriske værdier mellem kalibreringsfejl og regret
- ✓ én tabel med alle non-bo metrikker metrikker (rækker: surrogat, søjler: metrikker)
- ✓ act. improv. vs expected
- lokale metrikker: calibration på alle testpunkter inden for radius r af et træningspunkt

REAL

- opfølgende: gør GP mere eksplorativt
- akkumuleret regret: hvor god du er og hvor hurtig du er

MÅSKE

- gem også middelværdi for surrogaten i X[opt_idx]
- regn self.noise_var på stort sæt, plot sqrt(self.noise_var) sammen med regret # done
- Gem runtime for at se hastighed
