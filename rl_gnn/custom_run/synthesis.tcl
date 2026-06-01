yosys -import



plugin -i parmys

yosys -import



read_verilog -nomem2reg +/parmys/vtr_primitives.v

setattr -mod -set keep_hierarchy 1 single_port_ram

setattr -mod -set keep_hierarchy 1 dual_port_ram



puts "Using parmys as partial mapper"



# arch file: /home/digital-2/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml

# input files: [diffeq1.v]

# other args: [YYY]

# config file: /home/digital-2/vtr_exp/rl_gnn/custom_run/odin_config.xml

# output file: diffeq1.parmys.blif



parmys_arch -a /home/digital-2/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml



if {$env(PARSER) == "surelog" } {

	puts "Using Yosys read_uhdm command"

	plugin -i systemverilog

	yosys -import

	read_uhdm diffeq1.v

} elseif {$env(PARSER) == "system-verilog" } {

	puts "Using Yosys read_systemverilog command"

	plugin -i systemverilog

	yosys -import

	read_systemverilog diffeq1.v

} elseif {$env(PARSER) == "default" } {

	puts "Using Yosys read_verilog command"

	read_verilog -sv -nolatches diffeq1.v

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



parmys -a /home/digital-2/vtr_exp/rl_gnn/custom_run/vpr_arch_run.xml -nopass -c /home/digital-2/vtr_exp/rl_gnn/custom_run/odin_config.xml YYY



opt -full



techmap 

opt -fast



dffunmap

opt -fast -noff



#autoname



stat



hierarchy -check -auto-top -purge_lib



write_blif -true + vcc -false + gnd -undef + unconn -blackbox diffeq1.parmys.blif

