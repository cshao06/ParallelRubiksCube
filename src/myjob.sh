#!/bin/bash

# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=my_hw
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f19_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2


# Run your program
# (">" redirects the print output of your program,
#  in this case to "output.txt")

cd /home/zhitianx/term_project
./run > output.txt

