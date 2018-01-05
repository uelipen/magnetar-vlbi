  integer, parameter :: nsum=100,nsim=100
  real, dimension(nsum,nsim) :: chi2

  call random_seed()
  call random_number(chi2)
  chi2=-log(1-chi2)

  
