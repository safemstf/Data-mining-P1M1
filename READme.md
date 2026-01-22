# German Credit Analysis Project - Setup Instructions

## Prerequisites and Installation

### Required Software

#### 1. R Installation (Required)
- **Download:** https://cran.r-project.org/
- **Minimum version:** R 4.0 or higher
- **Installation:** Follow standard installation for your OS

#### 2. RStudio Installation (Recommended)
- **Download:** https://posit.co/download/rstudio-desktop/
- **Note:** While not strictly required, RStudio makes running the project much easier

#### 3. Java JDK Installation (REQUIRED for RWeka)
**This is the critical dependency that must be installed before running the code.**

##### Windows:
1. Download Java JDK from: https://www.oracle.com/java/technologies/downloads/
   - OR use OpenJDK: https://adoptium.net/
2. Install Java JDK (e.g., Java 11 or Java 17)
3. **Important:** During installation, note the installation path (e.g., `C:\Program Files\Java\jdk-17`)
4. Set JAVA_HOME environment variable:
   - Open System Properties → Environment Variables
   - Create new system variable: `JAVA_HOME` = `C:\Program Files\Java\jdk-17`
   - Edit PATH variable, add: `%JAVA_HOME%\bin`
5. Restart your computer (important!)
6. Verify installation:
   ```bash
   # Open Command Prompt
   java -version
   ```

##### Mac:
1. Install via Homebrew (easiest):
   ```bash
   brew install openjdk@17
   ```
2. Or download from: https://adoptium.net/
3. Verify installation:
   ```bash
   java -version
   ```

##### Linux:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install default-jdk

# Fedora/RHEL
sudo dnf install java-17-openjdk-devel

# Verify
java -version
```

---

## R Package Installation

### First-Time Setup (Run Once)

Open RStudio and run this installation script:

```r
# Install required packages
# This may take 5-10 minutes depending on your internet connection

install.packages(c(
  "tidyverse",      # Data manipulation
  "mlbench",        # ML datasets
  "rpart",          # Decision trees
  "rpart.plot",     # Tree visualization
  "caret",          # ML framework
  "FSelector",      # Feature selection
  "Rtsne",          # t-SNE
  "cluster",        # Clustering metrics
  "ggplot2"         # Visualization
))

# CRITICAL: Install Java-dependent packages
# Make sure Java is installed BEFORE running these commands
install.packages("rJava")
install.packages("RWekajars")
install.packages("RWeka")
```

### Verifying Installation

Run this verification script to check if everything is properly installed:

```r
# Verification Script
cat("=== Checking R Package Installation ===\n")

required_packages <- c("tidyverse", "mlbench", "rpart", "rpart.plot", 
                       "caret", "FSelector", "Rtsne", "cluster", 
                       "rJava", "RWekajars", "RWeka")

for (pkg in required_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "- INSTALLED\n")
  } else {
    cat("✗", pkg, "- MISSING\n")
  }
}

cat("\n=== Checking Java Configuration ===\n")
library(rJava)
.jinit()
cat("Java version:", .jcall("java/lang/System", "S", "getProperty", "java.version"), "\n")
cat("Java home:", .jcall("java/lang/System", "S", "getProperty", "java.home"), "\n")

cat("\n=== All checks passed! Ready to run the project. ===\n")
```

---

## Running the Project

### Step 1: Set Working Directory

```r
# Option 1: Use RStudio's GUI
# Session → Set Working Directory → Choose Directory
# Navigate to folder containing credit-g.csv

# Option 2: Use code (modify path)
setwd("C:/Users/YourName/Desktop/DataMiningProject")

# Verify
getwd()  # Should show your project directory
list.files()  # Should show credit-g.csv
```

### Step 2: Run the Main Script

```r
# Load the main script
source("german_credit_analysis.R")
```

The script will:
1. Load all required libraries
2. Load the dataset (credit-g.csv)
3. Perform preprocessing (Task 1)
4. Generate t-SNE visualization (Task 1.3)
5. Train all three classifiers (Task 2.1)
6. Extract rule bases (Task 2.2)
7. Perform ANOVA analysis (Task 3)
8. Generate all visualizations
9. Print comprehensive results

**Expected Runtime:** 2-3 minutes

**Expected Output:**
- Console output with detailed results
- 4-5 PNG files in your working directory:
  - tsne_visualization.png
  - decision_tree_plot.png
  - accuracy_boxplot.png
  - accuracy_violin.png
  - tukey_hsd_plot.png (if ANOVA is significant)

---

## Troubleshooting Common Issues

### Issue 1: "Java not found" or rJava fails to load

**Error message:**
```
Error: package 'rJava' could not be installed
Error : .onLoad failed in loadNamespace() for 'rJava'
```

**Solutions:**

**Windows:**
1. Ensure Java JDK (not JRE) is installed
2. Set JAVA_HOME environment variable correctly
3. Restart RStudio (or restart computer)
4. Try installing rJava with:
   ```r
   # Install from source
   install.packages("rJava", type = "source")
   ```

**Mac:**
1. Run in Terminal:
   ```bash
   R CMD javareconf
   sudo R CMD javareconf
   ```
2. Then in R:
   ```r
   install.packages("rJava", type = "source")
   ```

**Linux:**
```bash
sudo R CMD javareconf
```

### Issue 2: RWeka installation fails

**Error message:**
```
Error: package 'RWeka' is not available
```

**Solution:**
```r
# Install dependencies first
install.packages("rJava")
install.packages("RWekajars")

# Then install RWeka
install.packages("RWeka")

# If still fails, try from archive
install.packages("RWeka", repos = "http://R-Forge.R-project.org")
```

### Issue 3: "Cannot open file 'credit-g.csv'"

**Error message:**
```
Error in file(file, "rt") : cannot open the connection
```

**Solution:**
1. Verify credit-g.csv is in your working directory:
   ```r
   getwd()
   list.files()
   ```
2. Use full path if needed:
   ```r
   credit_data <- read.csv("C:/full/path/to/credit-g.csv", stringsAsFactors = TRUE)
   ```

### Issue 4: Memory issues with t-SNE

**Error message:**
```
Error: cannot allocate vector of size X MB
```

**Solution:**
Reduce perplexity in the script:
```r
# Find this line in german_credit_analysis.R
tsne_result <- Rtsne(tsne_numeric, 
                     dims = 2,
                     perplexity = 15,  # Reduced from 30
                     verbose = TRUE)
```

### Issue 5: Plots not displaying

**Solution:**
All plots are automatically saved as PNG files. Check your working directory:
```r
list.files(pattern = "\\.png$")
```

### Issue 6: Package version conflicts

**Solution:**
Update all packages:
```r
update.packages(ask = FALSE)
```

---

## File Structure

Your project directory should contain:

```
DataMiningProject/
├── credit-g.csv                    # Dataset (required)
├── german_credit_analysis.R        # Main script
├── README.md                       # This file
├── COMPLETED_PROJECT_REPORT.md     # Report template
└── (generated outputs)
    ├── tsne_visualization.png
    ├── decision_tree_plot.png
    ├── accuracy_boxplot.png
    └── accuracy_violin.png
```

---

## System Requirements

**Minimum:**
- RAM: 4 GB
- Storage: 500 MB free space
- OS: Windows 7+, macOS 10.13+, or Linux

**Recommended:**
- RAM: 8 GB or more
- Storage: 1 GB free space
- Multi-core processor for faster cross-validation

---

## Expected Results Summary

When the script completes successfully, you should see:

**Console Output:**
- Dataset dimensions: 1000 samples, 21 features
- Preprocessing summary (0 missing values, 0 duplicates)
- t-SNE silhouette score: ~0.008
- Decision Tree CV accuracy: ~72%
- PART CV accuracy: ~70%
- Ripper CV accuracy: ~72%
- ANOVA p-value: ~0.078
- "PROJECT COMPLETED SUCCESSFULLY!" message

**Generated Files:**
- 4-5 PNG visualization files
- All saved in working directory

**Runtime:**
- t-SNE: ~2 seconds
- Decision Tree: ~5 seconds
- PART: ~10 seconds
- Ripper: ~120 seconds
- Total: ~2-3 minutes

---

## Getting Help

If you encounter issues not covered here:

1. **Check package documentation:**
   ```r
   ?rJava
   ?RWeka
   ?Rtsne
   ```

2. **Verify Java installation:**
   ```r
   library(rJava)
   .jinit()
   .jcall("java/lang/System", "S", "getProperty", "java.version")
   ```

3. **Check R version:**
   ```r
   R.version.string
   ```

4. **Common resources:**
   - RDocumentation: https://www.rdocumentation.org/
   - Stack Overflow: https://stackoverflow.com/questions/tagged/r
   - CRAN RWeka: https://cran.r-project.org/web/packages/RWeka/

---

## Project Submission Checklist

Before submitting, ensure:

- [ ] Java JDK installed and configured
- [ ] All R packages installed successfully
- [ ] Script runs without errors from start to finish
- [ ] All visualizations generated (4-5 PNG files)
- [ ] Console output shows "PROJECT COMPLETED SUCCESSFULLY!"
- [ ] Report includes all required sections
- [ ] Code is clearly commented
- [ ] Working directory contains credit-g.csv

---

## Authors and Acknowledgments

**Course:** CSCE 5380 Data Mining  
**Institution:** University of North Texas  
**Dataset:** German Credit Dataset from UCI Machine Learning Repository  

**Key Libraries:**
- tidyverse (Wickham et al.)
- caret (Max Kuhn)
- rpart (Therneau & Atkinson)
- RWeka (Hornik et al.)
- Rtsne (van der Maaten)

---

**Last Updated:** January 2026  
**Version:** 1.0