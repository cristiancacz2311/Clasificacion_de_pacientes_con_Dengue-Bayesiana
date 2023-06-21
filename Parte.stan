data {
  int<lower=0> N;  // Número de observaciones
  int<lower=0> p;  // Número de variables explicativas
  int<lower=0, upper=1> y[N];  // Variable de respuesta binaria
  matrix[N, p] X;  // Matriz de diseño
}

parameters {
  vector[p] beta;  // Coeficientes del modelo
}

transformed parameters {
  vector[N] log_lik;  // Vector de log-likelihood
  
  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | X[i] * beta);
  }
}

model {
  // Prior para los coeficientes
  beta ~ normal(0, 100);
  
  // Modelo logístico
  target += sum(log_lik);
}
