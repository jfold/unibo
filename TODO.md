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

Perfekt kalibrering:

- Når sigma_n^2 er tæt på den sande OG posterior-variansens konfidensintervaller indeholder den underliggende f (ikke y!)
- Målingen lige nu er, om den prædiktive varians (posterior varians + sigma_n^2) indeholder y (testpunkterne)
- Har man real-world domæner, hvor man enten kan

  1. estimere sigma_n^2 og dermed f (fx mange samples for samme hyperparams.)
  2. har info omkring hvad f bør være
  3. ved at sigma_n^2 er lav
  4. ved at sigma_n^2 er høj: balancer/lær sharpness/calibration

- Hav to GP'er med hver sin kernel
- Sample fra den ene + støj
- Lav BO med de to GP'er, hvor de lærer hyperparams.
- Act. improv. (y_n+1 - y_n_so_far), (f_n+1 - f_n_so_far) vs. exp. improv.: for BO iteration
- Kalibrering på stort test set ift. f og y
