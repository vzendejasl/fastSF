#############################################################################################################################################
 # fastSF
 # 
 # Copyright (C) 2020, Mahendra K. Verma
 #
 # All rights reserved.
 # 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #     1. Redistributions of source code must retain the above copyright
 #        notice, this list of conditions and the following disclaimer.
 #     2. Redistributions in binary form must reproduce the above copyright
 #        notice, this list of conditions and the following disclaimer in the
 #        documentation and/or other materials provided with the distribution.
 #     3. Neither the name of the copyright holder nor the
 #        names of its contributors may be used to endorse or promote products
 #        derived from this software without specific prior written permission.
 # 
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 ############################################################################################################################################
 ##
 ##! \file test.py
 #
 #   \brief Script to generate plots for the test cases.
 #
 #   \author Shashwat Bhattacharya, Shubhadeep Sadhukhan
 #   \date Feb 2020
 #   \copyright New BSD License
 #
 ############################################################################################################################################
##
import yaml
import h5py
import os
import matplotlib
if "MPLBACKEND" not in os.environ:
	try:
		matplotlib.use("TkAgg")
	except Exception:
		matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from matplotlib import ticker, colors

mpl.style.use('classic')

plt.rcParams['xtick.major.size'] = 4.2
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 4.2
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
#plt.rcParams['axes.titlepad'] = 10

def resolve_repo_root(path_hint):
	path_hint = path_hint.strip()
	if not path_hint:
		return "."
	if os.path.isfile(path_hint):
		if os.path.basename(path_hint) == "para.yaml":
			parent = os.path.dirname(path_hint)
			return os.path.dirname(parent) if os.path.basename(parent) == "in" else parent
		return os.path.dirname(path_hint)
	if os.path.isfile(os.path.join(path_hint, "in", "para.yaml")):
		return path_hint
	if os.path.isfile(os.path.join(path_hint, "para.yaml")) and os.path.basename(path_hint) == "in":
		return os.path.dirname(path_hint)
	return path_hint


def cycle_label_from_path(path_hint):
	stem = os.path.splitext(os.path.basename(path_hint))[0]
	match = re.search(r"(\d+)$", stem)
	if not match:
		return "out"
	return f"structure_function_data_{int(match.group(1))}"


def resolve_output_dir(path_hint, repo_root):
	path_hint = path_hint.strip()
	if path_hint:
		if os.path.isfile(path_hint):
			return os.path.join(os.path.dirname(path_hint), cycle_label_from_path(path_hint))
		if os.path.isdir(path_hint):
			if os.path.isfile(os.path.join(path_hint, "SF_Grid_pll.h5")) or os.path.isfile(os.path.join(path_hint, "SF_Grid_scalar.h5")):
				return path_hint
			if os.path.isdir(os.path.join(path_hint, "out")):
				return os.path.join(path_hint, "out")
		candidate = os.path.join(repo_root, path_hint)
		if os.path.isdir(candidate):
			if os.path.isfile(os.path.join(candidate, "SF_Grid_pll.h5")) or os.path.isfile(os.path.join(candidate, "SF_Grid_scalar.h5")):
				return candidate
			if os.path.isdir(os.path.join(candidate, "out")):
				return os.path.join(candidate, "out")
	if os.path.isdir(os.path.join(repo_root, "out")):
		return os.path.join(repo_root, "out")
	return repo_root


def sf_file(filename):
	return os.path.join(output_dir, filename)


def save_plot(filename, dpi=600):
	plt.savefig(os.path.join(output_dir, filename), dpi=dpi)


def add_reference_constant_line(axes, r_plot, sf_plot, q_order):
	if q_order not in (2, 3):
		return
	valid = np.isfinite(sf_plot) & (sf_plot > 0)
	if not np.any(valid):
		return
	valid_values = sf_plot[valid]
	start = len(valid_values) // 3
	stop = max(start + 1, 2 * len(valid_values) // 3)
	ref_level = np.median(valid_values[start:stop])
	label = rf"Reference Const. (q={q_order})"
	axes.axhline(ref_level, color='black', lw=1.0, linestyle='dashed', label=label)


def parse_args():
	parser = argparse.ArgumentParser(description="Post-process fastSF HDF5 outputs.")
	parser.add_argument(
		"path",
		nargs="?",
		help="Path to the output folder or original input data file used by fastSF.",
	)
	parser.add_argument(
		"-r",
		"--repo-root",
		dest="repo_root",
		help="Path to the fastSF repo root or its in/ folder. Defaults to this script's repo.",
	)
	parser.add_argument(
		"-q",
		"--order",
		dest="order",
		help="Structure-function order to plot.",
	)
	return parser.parse_args()


cli_args = parse_args()
default_repo_root = os.path.dirname(os.path.abspath(__file__))

if cli_args.path is not None:
	input_path = cli_args.repo_root or default_repo_root
	data_path = cli_args.path
	q = cli_args.order if cli_args.order is not None else input("Enter the order: ")
else:
	input_path = cli_args.repo_root or input("Enter the path to your repo root or in folder: ")
	data_path = input("Enter the path to your output folder or input data file (blank = auto): ")
	q = cli_args.order if cli_args.order is not None else input("Enter the order: ")

repo_root = resolve_repo_root(input_path)
output_dir = resolve_output_dir(data_path, repo_root)

with open(os.path.join(repo_root, "in", "para.yaml"), 'r') as stream:
	try:
		para=(yaml.safe_load(stream))
		domain = para.get('domain_dimension', {})
		program = para.get('program', {})
		Lxx = domain.get('Lx', 1.0)
		Lyy = domain.get('Ly', 1.0)
		Lzz = domain.get('Lz', 1.0)
		scalar_switch = program.get('scalar_switch', False)
		two_dim_switch = program.get('2D_switch', False)
		longitudinal = program.get("Only_longitudinal", False)
		
	except yaml.YAMLError as exc:
		print(exc)
        	

def hdf5_reader(filename,dataset):
	file_V1_read = h5py.File(filename, 'r')
	dataset_V1_read = file_V1_read["/"+dataset]
	V1=dataset_V1_read[:,:,:]
	return V1

def hdf5_reader1D(filename,dataset):
	file_V1_read = h5py.File(filename, 'r')
	dataset_V1_read = file_V1_read["/"+dataset]
	V1=dataset_V1_read[:]
	return V1


def hdf5_reader_plane(filename,dataset):
	file_V1_read = h5py.File(filename, 'r')
	dataset_V1_read = file_V1_read["/"+dataset]
	V1=dataset_V1_read[:,:]
	return V1


def hdf5_reader_slice(filename,dataset, Ny):
	file_V1_read = h5py.File(filename)
	dataset_V1_read = file_V1_read["/"+dataset]
	V1=dataset_V1_read[:,Ny,:]
	return V1
A=9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)
B = 19





	
def plotSF_r_2D(data_path, q):
	SF = (hdf5_reader_plane(sf_file("SF_Grid_pll.h5"), "SF_Grid_pll"+str(q)))
	Nx, Nz = SF.shape
	
	Nr = int(np.ceil(np.sqrt((Nx-1)**2 + (Nz-1)**2)))+1

	r = np.zeros([Nr]) 
	for i in range(len(r)): #
		r[i]=np.sqrt(2)*i/(2*len(r))
	
	SF_r = np.zeros([Nr])
	
	counter = np.zeros([Nr])
	
	for x in range(Nx):
	    for z in range(Nz):
	        l = int(np.ceil(np.sqrt(x**2 + z**2)))
	        SF_r[l] = SF_r[l] + SF[x, z]
	        
	        counter[l] = counter[l] + 1	  
	  
	SF_r = SF_r/counter
	q_order = int(q)
	r_plot = r[1:len(r)]
	sf_plot = SF_r[1:len(r)]
	if q_order == 2:
		sf_plot = sf_plot * np.power(r_plot, -2.0/3.0)
		ylabel = r"$S_2^{u}(l)\, l^{-2/3}$"
		plot_label = r"$S_2^{u}(l)\, l^{-2/3}$"
		output_name = "SF_velocity_r2D_compensated_q2.png"
	elif q_order == 3:
		sf_plot = -sf_plot * np.power(r_plot, -1.0)
		ylabel = r"$-S_3^{u}(l)\, l^{-1}$"
		plot_label = r"$-S_3^{u}(l)\, l^{-1}$"
		output_name = "SF_velocity_r2D_compensated_q3.png"
	else:
		ylabel = r"$S_q^{u}(l)$"
		plot_label = rf"$S_{q_order}^{{u}}(l)$"
		output_name = "SF_velocity_r2D.png"
	
	fig, axes = plt.subplots(figsize = (3.5, 2.57))
	
	axes.plot(r_plot, sf_plot, color='red', lw=1.5, label=plot_label) 
	add_reference_constant_line(axes, r_plot, sf_plot, q_order)
	
	 
	axes.set_xlabel('$l$')
	axes.set_ylabel(ylabel)
	axes.set_xscale('log')
	if np.all(sf_plot > 0):
		axes.set_yscale('log')
	else:
		axes.set_yscale('symlog', linthresh=max(np.max(np.abs(sf_plot))*1e-6, 1e-12))
	fig.tight_layout()
	save_plot(output_name)
	


def plotSF_r_3D(data_path, q):
	SF = (hdf5_reader(sf_file("SF_Grid_pll.h5"), "SF_Grid_pll"+str(q)))
	
	
	Nx, Ny, Nz = SF.shape
	
	Nr = int(np.ceil(np.sqrt((Nx-1)**2 + (Ny-1)**2 + (Nz-1)**2)))+1
	

	r = np.zeros([Nr]) 
	for i in range(len(r)): 
		r[i]=np.sqrt(3)*i/(2*len(r))
	
	SF_r = np.zeros([Nr])
	
	counter = np.zeros([Nr])
	
	for x in range(Nx):
	    for y in range(Ny):
	        for z in range(Nz):
	           l = int(np.ceil(np.sqrt(x**2 + y**2 + z**2)))
	           SF_r[l] = SF_r[l] + SF[x, y, z]
	           
	           counter[l] = counter[l] + 1	  
	
	SF_r = SF_r/counter
	q_order = int(q)
	r_plot = r[1:len(r)]
	sf_plot = SF_r[1:len(r)]
	if q_order == 2:
		sf_plot = sf_plot * np.power(r_plot, -2.0/3.0)
		ylabel = r"$S_2^{u}(l)\, l^{-2/3}$"
		plot_label = r"$S_2^{u}(l)\, l^{-2/3}$"
		output_name = "SF_velocity_r3D_compensated_q2.png"
	elif q_order == 3:
		sf_plot = -sf_plot * np.power(r_plot, -1.0)
		ylabel = r"$-S_3^{u}(l)\, l^{-1}$"
		plot_label = r"$-S_3^{u}(l)\, l^{-1}$"
		output_name = "SF_velocity_r3D_compensated_q3.png"
	else:
		ylabel = r"$S_q^{u}(l)$"
		plot_label = rf"$S^{{u}}_{{\parallel,{q_order}}}(l)$"
		output_name = "SF_velocity_r3D.png"
	
	fig, axes = plt.subplots(figsize = (3.5, 2.57))
	
	axes.plot(r_plot, sf_plot, color='red', lw=1.5, label=plot_label)
	add_reference_constant_line(axes, r_plot, sf_plot, q_order)
	axes.set_xlabel('$l$')
	axes.set_ylabel(ylabel)
	axes.set_xscale('log')
	if np.all(sf_plot > 0):
		axes.set_yscale('log')
	else:
		axes.set_yscale('symlog', linthresh=max(np.max(np.abs(sf_plot))*1e-6, 1e-12))
	fig.tight_layout()
	save_plot(output_name)
		

	
def plot_SF2D_scalar(data_path, q):
    SF = (hdf5_reader_plane(sf_file("SF_Grid_scalar.h5"), "SF_Grid_scalar"+str(q)))
   
    
    Nlx, Nlz = SF.shape
    
    lx = np.linspace(0,Lxx/2.,Nlx)
    lz = np.linspace(0,Lzz/2.,Nlz)
    
    fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
    Lz,Lx=np.meshgrid(lz,lx)
    
    density = axes.contourf(lx, lz, np.transpose(SF), levels=np.linspace(SF.min(), SF.max(),50), cmap='jet')
    
    axes.set_xticks([0, Lxx/4., Lxx/2.])
    axes.set_yticks([0, Lzz/4., Lzz/2.])
    axes.set_xlabel('$l_x$')
    axes.set_ylabel('$l_z$')
    axes.tick_params(axis='x', which='major', pad=10)
    axes.tick_params(axis='y', which='major', pad=10)
    axes.set_title(r"$S^{\theta}$")

    
    axes.title.set_position([.5, 1.05])
    
    cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SF.min(), SF.max(),4))
    cb1.ax.tick_params(labelsize=A)
    
    fig.tight_layout()
    save_plot("SF_scalar2D.png")
    
   

	
def plot_SF2D_velocity(data_path, q):
    SFpll = (hdf5_reader_plane(sf_file("SF_Grid_pll.h5"), "SF_Grid_pll"+str(q)))
   
    
    Nlx, Nlz = SFpll.shape
    
    lx = np.linspace(0,Lxx/2.,Nlx)
    lz = np.linspace(0,Lzz/2.,Nlz)
    
    fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
  
    Lz,Lx=np.meshgrid(lz,lx)
   
    
    density = axes.contourf(lx, lz, np.transpose(SFpll), levels=np.linspace(SFpll.min(),SFpll.max(),50), cmap='jet')
    axes.set_xticks([0, Lxx/4., Lxx/2.])
    axes.set_yticks([0, Lzz/4., Lzz/2.])
    axes.set_xlabel('$l_x$')
    axes.set_ylabel('$l_z$')
    axes.tick_params(axis='x', which='major', pad=10)
    axes.tick_params(axis='y', which='major', pad=10)
    axes.set_title(r"$S^{u}_{\parallel}$")
    axes.title.set_position([.5, 1.05])
   
    cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SFpll.min(),SFpll.max(),4))#, ticks=[1e-4, 1e-2, 1e0]) ###### TICKS FOR THE COLORBARS ARE DEFINED HERE
    cb1.ax.tick_params(labelsize=A)
    
   
    fig.tight_layout()
    save_plot("SF_velocity2D_pll.png")
    
    if (longitudinal==False):
        SFperp = (hdf5_reader_plane(sf_file("SF_Grid_perp.h5"), "SF_Grid_perp"+str(q)))
        Nlx, Nlz = SFperp.shape
        lx = np.linspace(0,Lxx/2.,Nlx)
        lz = np.linspace(0,Lzz/2.,Nlz)
    
        fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
  
        Lz,Lx=np.meshgrid(lz,lx)
   
    
        density = axes.contourf(lx, lz, np.transpose(SFperp), levels=np.linspace(SFperp.min(),SFperp.max(),50), cmap='jet')
   
        axes.set_xticks([0, Lxx/4., Lxx/2.])
        axes.set_yticks([0, Lxx/4., Lzz/2.])
        axes.set_xlabel('$l_x$')
        axes.set_ylabel('$l_z$')
        axes.tick_params(axis='x', which='major', pad=10)
        axes.tick_params(axis='y', which='major', pad=10)
        axes.set_title(r"$S^{u}_{\perp}$")
        axes.title.set_position([.5, 1.05])
   
        cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SFperp.min(),SFperp.max(),4))#, ticks=[1e-4, 1e-2, 1e0]) ###### TICKS FOR THE COLORBARS ARE DEFINED HERE
        cb1.ax.tick_params(labelsize=A)
    
   
        fig.tight_layout()
        save_plot("SF_velocity2D_perp.png")
		
    
    
 

def plot_SF3D_scalar(data_path, q):
    SF = (hdf5_reader_slice(sf_file("SF_Grid_scalar.h5"), "SF_Grid_scalar"+str(q),-1))
   
    
    Nlx, Nlz= SF.shape

    lx = np.linspace(0, Lxx/2., Nlx)
    lz = np.linspace(0, Lzz/2., Nlz)
    
    
    fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
    
    Lz,Lx=np.meshgrid(lz,lx)
   

   
    
    
    density = axes.contourf(lx, lz, np.transpose(SF), levels=np.linspace(SF.min(),SF.max(),50), cmap='jet')
    
    
   
    axes.set_xticks([0, Lzz/4., Lxx/2.])
    axes.set_yticks([0, Lzz/4., Lzz/2.])
    axes.set_xlabel('$l_x$')
    axes.set_ylabel('$l_z$')
    axes.tick_params(axis='x', which='major', pad=10)
    axes.tick_params(axis='y', which='major', pad=10)
    axes.set_title(r"$S^{\theta}$")#, pad=10)

    
    axes.title.set_position([.5, 1.05])
    
    cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SF.min(),SF.max(),4))#, ticks=[1e-4, 1e-2, 1e0]) ###### TICKS FOR THE COLORBARS ARE DEFINED HERE
    cb1.ax.tick_params(labelsize=A)
    
    
    fig.tight_layout()
    save_plot("SF_scalar3D.png")
    
 
    
def plot_SF3D_velocity(data_path, q):
    SFpll = (hdf5_reader_slice(sf_file("SF_Grid_pll.h5"), "SF_Grid_pll"+str(q),-1))
    
    
    Nlx, Nlz= SFpll.shape

    lx = np.linspace(0, Lxx/2., Nlx)
    lz = np.linspace(0, Lzz/2., Nlz)
 
    Lz,Lx=np.meshgrid(lz,lx)
    fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
    
    density = axes.contourf(lx, lz, np.transpose(SFpll), levels=np.linspace(SFpll.min(),SFpll.max(),50), cmap='jet')
    
    axes.set_xticks([0, Lxx/4., Lxx/2.])
    axes.set_yticks([0, Lzz/4., Lzz/2.])
    axes.set_xlabel('$l_x$')
    axes.set_ylabel('$l_z$')
    axes.tick_params(axis='x', which='major', pad=10)
    axes.tick_params(axis='y', which='major', pad=10)
    axes.set_title(r"$S^{u}_{\parallel}$") #(l_x, l_y=0.5,l_z)$")#, pad=10)   
    axes.title.set_position([.5, 1.05])
    
    cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SFpll.min(),SFpll.max(),4))
    cb1.ax.tick_params(labelsize=A)
    
    
    fig.tight_layout()
    save_plot("SF_velocity3D_pll.png")
    if (longitudinal==False):
        SFperp = (hdf5_reader_slice(sf_file("SF_Grid_perp.h5"), "SF_Grid_perp"+str(q),-1))
        fig, axes = plt.subplots(1,1,figsize=(3.5,2.7),sharey=True)
        density = axes.contourf(lx, lz, np.transpose(SFperp), levels=np.linspace(SFperp.min(),SFperp.max(), 50), cmap='jet')
    
        axes.set_xticks([0, Lxx/4., Lxx/2.])
        axes.set_yticks([0, Lzz/4., Lzz/2.])
        axes.set_xlabel('$l_x$')
        axes.set_ylabel('$l_z$')
        axes.tick_params(axis='x', which='major', pad=10)
        axes.tick_params(axis='y', which='major', pad=10)
        axes.set_title(r"$S^{u}_{\perp}$") #(l_x, l_y=0.5,l_z)$")#, pad=10)   
        axes.title.set_position([.5, 1.05])
    
        cb1 = fig.colorbar(density, fraction=0.05, ax=axes,ticks=np.linspace(SFperp.min(),SFperp.max(),4))
        cb1.ax.tick_params(labelsize=A)
    
    
        fig.tight_layout()
        save_plot("SF_velocity3D_perp.png")
    
    
 





if scalar_switch:
	print ("Scalar")
	if two_dim_switch:
		print ("2D")
		plot_SF2D_scalar(data_path, q)
	else:
		print ("3D")
		plot_SF3D_scalar(data_path, q)
else:
	print ("Vector")
	if two_dim_switch:
		print ("2D")
		plot_SF2D_velocity(data_path, q)
		plotSF_r_2D(data_path,2)
	else:
		print ("3D")
		plot_SF3D_velocity(data_path, q)
		plotSF_r_3D(data_path,q)

plt.show()
