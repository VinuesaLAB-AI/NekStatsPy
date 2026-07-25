
INC_SRC := ./inc_src
$(OBJDIR)/misc_outgrid.o     :./misc_outgrid.f;                      $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@

