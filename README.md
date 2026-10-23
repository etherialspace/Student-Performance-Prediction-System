# Intelligent Student Performance Prediction System

## Objective
Develop an AI system that predicts student performance based on academic information using **Machine Learning**, **Classification**, and a **Decision Tree** algorithm, and provides targeted diagnostic suggestions for areas where the student needs improvement.

## Features & Tasks Implemented
1. **Student Input Collection**: Accepts student name, roll/ID, and 4 key academic metrics:
   - **Attendance (%)** (`0 - 100`)
   - **Assignment Marks** (`0 - 100`)
   - **Internal Marks** (`0 - 100`)
   - **Study Hours per Day** (`0 - 16`)
2. **Decision Tree Classification**: Classifies performance into one of four tiers:
   - `Excellent`
   - `Good`
   - `Average`
   - `Poor`
3. **Advanced Feature — Diagnostic Improvement Engine**: Pinpoints specific weak areas (with priority levels: `Critical`, `Moderate`, `Minor`), compares current values against recommended benchmarks, and generates a personalized study & action plan.

## Project Structure
- `student_performance_predictor.py` — Main AI prediction script (uses `scikit-learn`'s `DecisionTreeClassifier`, with a built-in CART Gini Decision Tree fallback so it runs even without external packages installed).
- `student_dataset.csv` — Training dataset of 56 student academic records across all four performance tiers.
- `requirements.txt` — Optional Python package dependencies (`scikit-learn`).

## How to Run

### 1. Interactive CLI Mode
```bash
python3 student_performance_predictor.py
```

### 2. Demo Mode (Runs sample predictions for Excellent, Good, Average, and Poor tiers)
```bash
python3 student_performance_predictor.py --demo
```
