yosys -import

plugin -i parmys



# yosys-slang plugin error handling

if {$env(PARSER) == "slang" } {

	if {![info exists ::env(yosys_slang_path)]} {

		puts "Error: $err"

		puts "yosys_slang_path is not set"

	} elseif {![file exists $::env(yosys_slang_path)]} {

		error "Error: cannot find plugin at '$::env(yosys_slang_path)'. Run make with CMake param -DSLANG_SYSTEMVERILOG=ON to enable yosys-slang plugin."

	} else {

		plugin -i slang

		yosys -import

		puts "Using yosys-slang as yosys frontend"

	}

} elseif {$env(PARSER) == "default" } {

	yosys -import

	puts "Using Yosys read_verilog as yosys frontend"

} else {

	error "Invalid PARSER"

}



# arch file: /root/desktop/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml

# input files: [diffeq2.v]

# other args: []

# config file: /root/desktop/vtr_exp/rl_gnn/custom_run/odin_config.xml

# output file: diffeq2.parmys.blif



parmys_arch -a /root/desktop/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml



if {$env(PARSER) == "slang" } {

	# Create a file list containing the name(s) of file(s) \

	# to read together with read_slang

	source [file join [pwd] "slang_filelist.tcl"]

	set readfile [file join [pwd] "filelist.txt"]

	#Writing names of circuit files to file list

	build_filelist {diffeq2.v} $readfile

	puts "Using Yosys read_slang command"

	#Read vtr_primitives library and user design verilog in same command

	read_slang -v $env(PRIMITIVES) -C $readfile

} elseif {$env(PARSER) == "default" } {

	puts "Using Yosys read_verilog command"

	read_verilog -nomem2reg +/parmys/vtr_primitives.v

	setattr -mod -set keep_hierarchy 1 single_port_ram

 	setattr -mod -set keep_hierarchy 1 dual_port_ram

	read_verilog -sv -nolatches diffeq2.v

} else {

	error "Invalid PARSER"

}



# Check that there are no combinational loops

scc -select

select -assert-none %

select -clear



hierarchy -check -auto-top -purge_lib



opt_expr

opt_clean

check

opt -nodffe -nosdff

procs -norom

fsm

opt

wreduce

peepopt

opt_clean

share



opt -full

memory -nomap

flatten



opt -full



techmap -map +/parmys/adff2dff.v

techmap -map +/parmys/adffe2dff.v

techmap -map +/parmys/aldff2dff.v

techmap -map +/parmys/aldffe2dff.v



opt -full



# Separate options for Parmys execution (Verilog or SystemVerilog)

if {$env(PARSER) == "default" || $env(PARSER) == "slang"} {

    # For Verilog, use -nopass for a simpler, faster flow

    parmys -a /root/desktop/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml -nopass -c /root/desktop/vtr_exp/rl_gnn/custom_run/odin_config.xml 

} 



opt -full



techmap 

opt -fast



dffunmap

opt -fast -noff

#autoname



stat



hierarchy -check -auto-top -purge_lib



write_blif -true + vcc -false + gnd -undef + unconn -blackbox diffeq2.parmys.blif

