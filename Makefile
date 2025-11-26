# =========================================================
# Engression Simulation Study — Makefile Automation
# Repo root: engression_python/
# =========================================================

PYTHON := python3

# Paths
RESULT_DIR := result
FIG_DIR := $(RESULT_DIR)/figures

# CSV outputs
CSV_BASE := $(RESULT_DIR)/mse_engression_losses.csv
CSV_OPT  := $(RESULT_DIR)/mse_engression_losses_optimized.csv

# Scripts
SIM_SCRIPT       := test.py
SIM_OPT_SCRIPT   := test_optimized.py
BOXPLOT_SCRIPT   := plot_results.py
DIST_SCRIPT      := plot_loss_distributions.py
PROFILE_SCRIPT   := complexity_profile.py
PROFILE_OPT_SCRIPT := complexity_profile_optimized.py
BOTTLENECK_SCRIPT  := bottleneck.py
PERF_SCRIPT        := plot_performance_comparison.py
REGRESSION_SCRIPT  := check_regression_mse.py


# ---------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------
.PHONY: all simulate simulate_opt figures distributions boxplots analysis \
        clean help profile profile_opt bottleneck perfplots regression_test

all: simulate analysis


# ---------------------------------------------------------
# Simulation Targets
# ---------------------------------------------------------
simulate: $(CSV_BASE)

simulate_opt: $(CSV_OPT)


# Baseline (serial)
$(CSV_BASE):
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(SIM_SCRIPT)
	@test -f $(CSV_BASE) || (echo "[ERROR] Expected $(CSV_BASE) not found." && exit 1)

# Optimized (parallel)
$(CSV_OPT):
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(SIM_OPT_SCRIPT)
	@test -f $(CSV_OPT) || (echo "[ERROR] Expected $(CSV_OPT) not found." && exit 1)


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
analysis: boxplots distributions
figures: boxplots distributions

boxplots: $(FIG_DIR)/.ok_boxplots
distributions: $(FIG_DIR)/.ok_distributions


$(FIG_DIR)/.ok_boxplots: $(CSV_BASE) $(BOXPLOT_SCRIPT)
	@mkdir -p $(FIG_DIR)
	$(PYTHON) $(BOXPLOT_SCRIPT)
	@touch $@

$(FIG_DIR)/.ok_distributions: $(CSV_BASE) $(DIST_SCRIPT)
	@mkdir -p $(FIG_DIR)
	$(PYTHON) $(DIST_SCRIPT)
	@touch $@


# ---------------------------------------------------------
# Profiling
# ---------------------------------------------------------
profile:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(PROFILE_SCRIPT)
	@echo "[INFO] Baseline profiling saved to $(RESULT_DIR)/complexity_grid.csv"

profile_opt:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(PROFILE_OPT_SCRIPT)
	@echo "[INFO] Optimized profiling saved to $(RESULT_DIR)/complexity_grid_optimized.csv"


# ---------------------------------------------------------
# Bottleneck Timing
# ---------------------------------------------------------
bottleneck:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(BOTTLENECK_SCRIPT)
	@echo "[INFO] Bottleneck timings saved to $(RESULT_DIR)/bottleneck_times.csv"


# ---------------------------------------------------------
# Performance Comparison (Runtime Visualization)
# ---------------------------------------------------------
perfplots:
	@mkdir -p $(FIG_DIR)
	$(PYTHON) $(PERF_SCRIPT)
	@echo "[INFO] Performance plots saved to $(FIG_DIR)"


# ---------------------------------------------------------
# Regression Test (Baseline vs Optimized MSE)
# ---------------------------------------------------------
regression_test:
	$(PYTHON) $(REGRESSION_SCRIPT)


# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------
clean:
	@echo "🧹 Cleaning generated figures and summaries (keeps the main CSVs)..."
	@rm -f $(FIG_DIR)/*.png
	@rm -f $(FIG_DIR)/.ok_*
	@rm -f $(RESULT_DIR)/high_mse_cases.csv
	@rm -f $(RESULT_DIR)/high_mse_rows.csv
	@rm -f $(RESULT_DIR)/high_mse_counts_by_loss.csv


# ---------------------------------------------------------
# Help
# ---------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make simulate          # run baseline (serial) simulation"
	@echo "  make simulate_opt      # run optimized parallel simulation"
	@echo "  make boxplots          # generate boxplots"
	@echo "  make distributions     # generate violin/density/ECDF plots"
	@echo "  make figures           # all plots"
	@echo "  make analysis          # analysis pipeline"
	@echo "  make profile           # baseline profiling"
	@echo "  make profile_opt       # optimized profiling"
	@echo "  make bottleneck        # bottleneck timing"
	@echo "  make perfplots         # baseline vs optimized runtime plots"
	@echo "  make regression_test   # check correctness baseline vs optimized"
	@echo "  make clean             # remove figures/summaries"
	@echo "  make all               # simulate + analysis"
