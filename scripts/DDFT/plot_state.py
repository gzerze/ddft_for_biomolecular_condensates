import argparse
from Protein_RPA.utils.input_parse import input_parse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm,ticker
from postprocess import *
import fipy as fp

from os import listdir
from os.path import join, isfile

#def plot_densitylog_2D(mesh,phi,t,steps,output_dir):
#	fig,ax=plt.subplots()
#	cs = ax.contourf(mesh.x.globalValue,mesh.y.globalValue,gv.r_res*phi.globalValue,cmap=cm.get_cmap("Blues"),levels=np.linspace(0,0.2,256),locator=ticker.LogLocator())	
#	fig.colorbar(cs)
#	ax.legend(title=str(int(t)))
 #   ax.set_xlabel('y')
#	ax.set_ylabel('x')
#	plt.title("Volume fraction")
#	fig.savefig(fname=output_dir+'Plog_step_{step}.png'.format(step=steps),dpi=300,format='png')


def read_state(args):
    input_state=args.s
    output_dir=args.o
    input_param=args.i

    input_parameters = input_parse(args.i);

    nx = int(input_parameters['nx'])
    ny = int(input_parameters['ny'])
    dx = input_parameters['dx']
    phis=input_parameters['phis']
    kappa=input_parameters['kappa']
    u=input_parameters['temperature_inverse']

    mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dx)
    mesh=mesh-float(nx)*dx*0.5


    phi_range=[0,0.25]
    xi_range=[-2,2]
    phi=fp.CellVariable(mesh=mesh, name=r'$\phi_{monomer}$')
    xi = fp.CellVariable(mesh=mesh)
    print(listdir(input_state))

    for filename in listdir(input_state):
        full_path = join(input_state, filename)
        state_file=open(full_path,'rb')
        steps,t,input_parameters,phi_value,xi_value=pickle.load(state_file) 
        print(t)
        state_file.close()
        steps+=1
 
        phi.setValue(phi_value.flatten())
        xi.setValue(xi_value.flatten())

        #plot_densitylog_2D(mesh,phi,t,steps,output_dir)
        plot_density_2D(mesh,phi,t,steps,output_dir,phi_range)
        plot_density_1D(mesh,phi_value,t,steps,output_dir,phi_range)
        plot_chempotential(mesh,xi,t,steps,output_dir,xi_range)

        del phi_value
        del xi_value

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description='Take output filename to produce Movie')
     parser.add_argument('--o',help="output of plot density", required = True);
     parser.add_argument('--i',help="input parameters", required = True);
     
     parser.add_argument('--s',help="Directory of state files", required = True);
     args = parser.parse_args();

     read_state(args);
