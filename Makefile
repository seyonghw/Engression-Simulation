# =========================================================
# Engression Simulation Study — Makefile Automation
# Repo root: engression_python/
# =========================================================

PYTHON := python3

# Paths
RESULT_DIR := result
FIG_DIR := $(RESULT_DIR)/figures
CSV := $(RESULT_DIR)/mse_engression_losses.csv

# Scripts
SIM_SCRIPT := test.py
BOXPLOT_SCRIPT := plot_results.py
DIST_SCRIPT := plot_loss_distributions.py
PROFILE_SCRIPT := complexity_profile.py
PROFILE_OPT_SCRIPT := complexity_profile_optimized.py
BOTTLENECK_SCRIPT := bottleneck.py

# ---------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------
.PHONY: all simulate figures distributions boxplots analysis clean help profile

all: simulate analysis

simulate: $(CSV)

analysis: boxplots distributions

figures: boxplots distributions

boxplots: $(FIG_DIR)/.ok_boxplots
distributions: $(FIG_DIR)/.ok_distributions

# ---------------------------------------------------------
# File targets
# ---------------------------------------------------------
# 1) Run the full simulation to produce the main CSV
$(CSV):
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(SIM_SCRIPT)
	@test -f $(CSV) || (echo "[ERROR] Expected $(CSV) not found. Ensure test.py writes to this path." && exit 1)

# 2) Boxplots of case-level mean MSE (per (model, dx,dy,k))
$(FIG_DIR)/.ok_boxplots: $(CSV) $(BOXPLOT_SCRIPT)
	@mkdir -p $(FIG_DIR)
	$(PYTHON) $(BOXPLOT_SCRIPT)
	@touch $@

# 3) Distribution comparisons (violin/density/ECDF) for 4 losses
$(FIG_DIR)/.ok_distributions: $(CSV) $(DIST_SCRIPT)
	@mkdir -p $(FIG_DIR)
	$(PYTHON) $(DIST_SCRIPT)
	@touch $@

# 4) Profiling / complexity runs over (REPS, N_TRAIN) grid
profile:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(PROFILE_SCRIPT)
	@echo "[INFO] Profiling / complexity results should be in $(RESULT_DIR)/complexity_grid.csv"

# 5) Profiling / complexity runs — optimized parallel version
profile_opt:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(PROFILE_OPT_SCRIPT)
	@echo "[INFO] Optimized profiling results should be in $(RESULT_DIR)/complexity_grid_optimized.csv"

# 6) Bottleneck timing for internal steps (generate_mats, data gen, training, prediction)
bottleneck:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) $(BOTTLENECK_SCRIPT)
	@echo "[INFO] Bottleneck timings should be in $(RESULT_DIR)/bottleneck_times.csv"



# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
clean:
	@echo "🧹 Cleaning generated figures and summaries (keeps the main CSV)..."
	@rm -f $(FIG_DIR)/*.png
	@rm -f $(FIG_DIR)/.ok_*
	@rm -f $(RESULT_DIR)/high_mse_cases.csv
	@rm -f $(RESULT_DIR)/high_mse_rows.csv
	@rm -f $(RESULT_DIR)/high_mse_counts_by_loss.csv

help:
	@echo "Targets:"
	@echo "  make simulate     # run the 72x10 simulation and write $(CSV)"
	@echo "  make boxplots     # build boxplots per (model, dx,dy,k)"
	@echo "  make distributions# build violin/density/ECDF comparisons"
	@echo "  make figures      # boxplots + distributions"
	@echo "  make analysis     # figures"
	@echo "  make profile       # run profiling / complexity grid (profile.py)"
	@echo "  make profile_opt     # run optimized profiling (complexity_profile_optimized.py)"
	@echo "  make bottleneck      # run bottleneck timing (bottleneck.py)"
	@echo "  make all          # simulate + analysis"
	@echo "  make clean        # remove figures and summaries (keep CSV)"
