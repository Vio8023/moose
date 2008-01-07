//moose
// This simulation tests readcell formation of a compartmental model
// of an axon.

float SIMDT = 1e-5
float PLOTDT = 1e-4
float RUNTIME = 0.05
float INJECT = 1e-9

// settab2const sets a range of entries in a tabgate table to a constant
function settab2const(gate, table, imin, imax, value)
	str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
		setfield {gate} {table}->table[{i}] {value} 
	end
end

addalias setup_table2 setupgate
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau
addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau

include bulbchan.g

create neutral /library
create compartment /library/compartment

ce /library
make_K_mit_usb
make_Na_mit_usb
ce /

readcell soma.p /axon

create table /Vm0
call /Vm0 TABCREATE {RUNTIME / PLOTDT} 0 {RUNTIME}
setfield /Vm0 step_mode 3
addmsg /axon/soma /Vm0 INPUT Vm

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {PLOTDT}

useclock /axon/##[TYPE=compartment],/axon/##[TYPE=tabchannel] 0
useclock /axon/# 1 init
useclock /Vm0 2

reset
setfield /axon/soma inject {INJECT}
step {RUNTIME} -t

tab2file axon0.plot /Vm0 table
