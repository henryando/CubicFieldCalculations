The current goal of this project is to do cubic crystal field modeling.

CubicFields.py contains a number of functions relating to this task, but at the top level the only one you need is

energies(J, W, x)

which returns a list of the eigenvalues of the Hamiltonian described by the total angular momentum J, the scale parameter W, and the mixing parameter x. W and x come from Lea Leask and Wolf. 
