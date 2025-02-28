# 
# Synthesis run script generated by Vivado
# 

set TIME_start [clock seconds] 
proc create_report { reportName command } {
  set status "."
  append status $reportName ".fail"
  if { [file exists $status] } {
    eval file delete [glob $status]
  }
  send_msg_id runtcl-4 info "Executing : $command"
  set retval [eval catch { $command } msg]
  if { $retval != 0 } {
    set fp [open $status w]
    close $fp
    send_msg_id runtcl-5 warning "$msg"
  }
}
set_param synth.elaboration.rodinMoreOptions {rt::set_parameter ignoreVhdlAssertStmts false}
create_project -in_memory -part xcu280-fsvh2892-2L-e

set_param project.singleFileAddWarning.threshold 0
set_param project.compositeFile.enableAutoGeneration 0
set_param synth.vivado.isSynthRun true
set_property webtalk.parent_dir /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/vivadocompile/vivadocompile.cache/wt [current_project]
set_property parent.project_path /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/vivadocompile/vivadocompile.xpr [current_project]
set_property default_lib xil_defaultlib [current_project]
set_property target_language Verilog [current_project]
set_property ip_cache_permissions {read write} [current_project]
read_verilog -library xil_defaultlib {
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N0.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N1.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N10.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N11.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N12.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N13.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N14.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N15.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N16.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N17.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N18.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N19.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N2.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N20.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N21.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N22.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N23.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N24.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N25.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N26.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N27.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N28.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N29.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N3.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N30.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N31.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N4.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N5.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N6.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N7.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N8.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer0_N9.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N0.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N1.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N10.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N11.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N12.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N13.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N14.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N15.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N2.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N3.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N4.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N5.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N6.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N7.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N8.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/layer1_N9.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/myreg.v
  /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/logicnet.v
}
# Mark all dcp files as not used in implementation to prevent them from being
# stitched into the results of this synthesis run. Any black boxes in the
# design are intentionally left as such for best results. Dcp files will be
# stitched into the design at a later time, either when this synthesis run is
# opened, or when it is stitched into a dependent implementation run.
foreach dcp [get_files -quiet -all -filter file_type=="Design\ Checkpoint"] {
  set_property used_in_implementation false $dcp
}
read_xdc /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/logicnet.xdc
set_property used_in_implementation false [get_files /workspace/logicnets/examples/hgcal_autoencoder/pipecleaner/hgcal_quant2/results_logicnet/logicnet.xdc]

set_param ips.enableIPCacheLiteLoad 1
close [open __synthesis_is_running__ w]

synth_design -top logicnet -part xcu280-fsvh2892-2L-e -fanout_limit 400 -directive PerformanceOptimized -retiming -fsm_extraction one_hot -keep_equivalent_registers -resource_sharing off -no_lc -shreg_min_size 5 -mode out_of_context


# disable binary constraint mode for synth run checkpoints
set_param constraints.enableBinaryConstraints false
write_checkpoint -force -noxdef logicnet.dcp
create_report "synth_1_synth_report_utilization_0" "report_utilization -file logicnet_utilization_synth.rpt -pb logicnet_utilization_synth.pb"
file delete __synthesis_is_running__
close [open __synthesis_is_complete__ w]
