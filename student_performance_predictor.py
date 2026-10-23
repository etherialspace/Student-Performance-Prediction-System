"""
Intelligent Student Performance Prediction System
=================================================
Objective:
    Develop an AI system that predicts student performance based on academic
    information using Machine Learning (Decision Tree Classification) and
    provides targeted suggestions for areas needing improvement.

Features Used:
    1. Attendance (%)          [0 - 100]
    2. Assignment Marks        [0 - 100]
    3. Internal Marks          [0 - 100]
    4. Study Hours (per day)   [0 - 12]

Target Classes:
    - Excellent
    - Good
    - Average
    - Poor
"""

import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

# Attempt to import scikit-learn DecisionTreeClassifier.
# Provides a transparent CART Decision Tree fallback if scikit-learn is not installed
# in the active Python environment.
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# Fallback CART Decision Tree (used only if scikit-learn is not installed)
# ============================================================================
class _Node:
    def __init__(
        self,
        predicted_class: str,
        class_counts: Dict[str, int],
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ):
        self.predicted_class = predicted_class
        self.class_counts = class_counts
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class FallbackDecisionTreeClassifier:
    """CART Decision Tree Classifier using Gini Impurity (mirrors scikit-learn API)."""

    def __init__(self, max_depth: int = 5, random_state: int = 42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.classes_: List[str] = []
        self.feature_importances_: List[float] = []
        self.tree_: Optional[_Node] = None

    @staticmethod
    def _gini(labels: List[str]) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        counts: Dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def fit(self, X: List[List[float]], y: List[str]) -> "FallbackDecisionTreeClassifier":
        self.classes_ = sorted(list(set(y)))
        n_features = len(X[0])
        importances = [0.0] * n_features
        self.tree_ = self._grow_tree(X, y, depth=0, importances=importances)
        total_imp = sum(importances)
        if total_imp > 0:
            self.feature_importances_ = [imp / total_imp for imp in importances]
        else:
            self.feature_importances_ = [1.0 / n_features] * n_features
        return self

    def _grow_tree(
        self,
        X: List[List[float]],
        y: List[str],
        depth: int,
        importances: List[float],
    ) -> _Node:
        counts: Dict[str, int] = {cls: 0 for cls in self.classes_}
        for lbl in y:
            counts[lbl] = counts.get(lbl, 0) + 1
        predicted_class = max(counts, key=lambda k: counts[k])
        node = _Node(predicted_class=predicted_class, class_counts=counts)

        if depth < self.max_depth and len(set(y)) > 1 and len(y) >= 2:
            best_feat, best_thresh, best_gain = self._best_split(X, y)
            if best_feat is not None and best_thresh is not None and best_gain > 0:
                left_X, left_y, right_X, right_y = [], [], [], []
                for row, lbl in zip(X, y):
                    if row[best_feat] <= best_thresh:
                        left_X.append(row)
                        left_y.append(lbl)
                    else:
                        right_X.append(row)
                        right_y.append(lbl)
                if left_y and right_y:
                    importances[best_feat] += best_gain * len(y)
                    node.feature_index = best_feat
                    node.threshold = best_thresh
                    node.left = self._grow_tree(left_X, left_y, depth + 1, importances)
                    node.right = self._grow_tree(right_X, right_y, depth + 1, importances)
        return node

    def _best_split(
        self, X: List[List[float]], y: List[str]
    ) -> Tuple[Optional[int], Optional[float], float]:
        n = len(y)
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        n_features = len(X[0])

        for feat_idx in range(n_features):
            values = sorted(set(row[feat_idx] for row in X))
            thresholds = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
            for thresh in thresholds:
                left_y = [y[i] for i in range(n) if X[i][feat_idx] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat_idx] > thresh]
                if not left_y or not right_y:
                    continue
                weighted_gini = (
                    len(left_y) / n * self._gini(left_y)
                    + len(right_y) / n * self._gini(right_y)
                )
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _predict_one(self, row: List[float]) -> _Node:
        node = self.tree_
        while node and node.feature_index is not None and node.threshold is not None:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node  # type: ignore[return-value]

    def predict(self, X: List[List[float]]) -> List[str]:
        return [self._predict_one(row).predicted_class for row in X]

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        probas = []
        for row in X:
            leaf = self._predict_one(row)
            total = sum(leaf.class_counts.values()) or 1
            probas.append([leaf.class_counts.get(cls, 0) / total for cls in self.classes_])
        return probas

    def export_rules(self, feature_names: List[str]) -> str:
        lines: List[str] = []

        def _recurse(node: Optional[_Node], indent: str = "") -> None:
            if node is None:
                return
            if node.feature_index is None or node.threshold is None:
                lines.append(f"{indent}|--- class: {node.predicted_class}")
                return
            fname = feature_names[node.feature_index]
            lines.append(f"{indent}|--- {fname} <= {node.threshold:.2f}")
            _recurse(node.left, indent + "|   ")
            lines.append(f"{indent}|--- {fname} >  {node.threshold:.2f}")
            _recurse(node.right, indent + "|   ")

        _recurse(self.tree_)
        return "\n".join(lines)


# ============================================================================
# Dataset Generation & Loading
# ============================================================================
FEATURE_NAMES = ["Attendance", "Assignment_Marks", "Internal_Marks", "Study_Hours"]

# Representative training records:
# [Attendance (%), Assignment Marks (/100), Internal Marks (/100), Study Hours (/day), Performance]
# Includes diverse profiles so the Decision Tree learns splits across all 4 features.
DEFAULT_TRAINING_DATA = [
    # Excellent Students (strong across internal marks, study hours, attendance, assignments)
    [96, 94, 92, 8.5, "Excellent"],
    [92, 90, 89, 7.5, "Excellent"],
    [95, 92, 94, 8.0, "Excellent"],
    [88, 87, 86, 7.0, "Excellent"],
    [91, 89, 90, 7.5, "Excellent"],
    [98, 96, 95, 9.0, "Excellent"],
    [84, 91, 88, 7.5, "Excellent"],
    [94, 88, 91, 8.0, "Excellent"],
    [85, 89, 83, 7.8, "Excellent"],
    [93, 95, 93, 8.5, "Excellent"],
    [86, 88, 86, 6.8, "Excellent"],
    [97, 93, 96, 9.2, "Excellent"],
    [82, 90, 84, 7.8, "Excellent"],
    [90, 86, 88, 6.9, "Excellent"],

    # Good Students (including high internal marks or study hours with moderate other metrics)
    [92, 78, 76, 5.2, "Good"],
    [80, 75, 74, 5.0, "Good"],
    [85, 80, 79, 6.0, "Good"],
    [78, 76, 75, 5.0, "Good"],
    [89, 74, 78, 5.5, "Good"],
    [76, 74, 72, 4.8, "Good"],
    [83, 86, 77, 5.8, "Good"],
    [79, 77, 73, 5.2, "Good"],
    [74, 78, 86, 4.5, "Good"],
    [77, 72, 74, 4.5, "Good"],
    [88, 90, 78, 7.2, "Good"],
    [75, 81, 85, 4.8, "Good"],
    [86, 79, 81, 6.6, "Good"],
    [88, 88, 79, 5.4, "Good"],

    # Average Students (including borderline internal marks / study hours with lower assignment or attendance)
    [85, 62, 60, 3.2, "Average"],
    [65, 58, 57, 3.0, "Average"],
    [78, 65, 63, 3.5, "Average"],
    [60, 55, 54, 2.8, "Average"],
    [70, 64, 62, 4.8, "Average"],
    [82, 60, 58, 2.9, "Average"],
    [67, 63, 61, 3.5, "Average"],
    [66, 62, 73, 3.2, "Average"],
    [62, 56, 55, 3.0, "Average"],
    [68, 60, 72, 3.1, "Average"],
    [66, 59, 60, 3.1, "Average"],
    [75, 67, 64, 3.9, "Average"],
    [64, 72, 59, 3.3, "Average"],
    [71, 49, 65, 3.6, "Average"],

    # Poor Students (including borderline internal marks / study hours with very low attendance / assignments)
    [45, 40, 38, 1.5, "Poor"],
    [52, 45, 42, 2.0, "Poor"],
    [68, 42, 39, 1.5, "Poor"],
    [55, 48, 44, 2.2, "Poor"],
    [42, 35, 34, 1.2, "Poor"],
    [62, 44, 40, 1.8, "Poor"],
    [35, 28, 25, 0.8, "Poor"],
    [48, 53, 44, 1.5, "Poor"],
    [54, 44, 55, 2.9, "Poor"],
    [40, 30, 32, 1.0, "Poor"],
    [47, 41, 39, 1.6, "Poor"],
    [58, 46, 43, 2.1, "Poor"],
    [50, 39, 37, 1.4, "Poor"],
    [52, 45, 41, 3.0, "Poor"],
]


def ensure_dataset_csv(csv_path: str) -> None:
    """Create the dataset CSV file if it does not already exist."""
    if os.path.exists(csv_path):
        return
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["Performance"])
        writer.writerows(DEFAULT_TRAINING_DATA)


def load_dataset(csv_path: str) -> Tuple[List[List[float]], List[str]]:
    """Load training features X and target labels y from the CSV file."""
    ensure_dataset_csv(csv_path)
    X: List[List[float]] = []
    y: List[str] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([
                float(row["Attendance"]),
                float(row["Assignment_Marks"]),
                float(row["Internal_Marks"]),
                float(row["Study_Hours"]),
            ])
            y.append(row["Performance"].strip())
    return X, y


# ============================================================================
# Model Training & Prediction Engine
# ============================================================================
class StudentPerformanceSystem:
    """
    AI System that trains a Decision Tree Classifier on student academic records,
    predicts student performance, and generates personalized improvement plans.
    """

    # Ideal academic benchmarks used by the diagnostic advisor
    BENCHMARKS = {
        "Attendance": {"min_good": 75.0, "excellent": 88.0, "unit": "%"},
        "Assignment_Marks": {"min_good": 70.0, "excellent": 85.0, "unit": "/100"},
        "Internal_Marks": {"min_good": 70.0, "excellent": 85.0, "unit": "/100"},
        "Study_Hours": {"min_good": 4.5, "excellent": 7.0, "unit": " hrs/day"},
    }

    def __init__(self, csv_path: str = "student_dataset.csv"):
        self.csv_path = csv_path
        self.X, self.y = load_dataset(self.csv_path)
        if SKLEARN_AVAILABLE:
            self.model = DecisionTreeClassifier(
                criterion="gini", max_depth=4, random_state=42
            )
        else:
            self.model = FallbackDecisionTreeClassifier(max_depth=4, random_state=42)
        self.accuracy: float = 0.0
        self._train_and_evaluate()

    def _train_and_evaluate(self) -> None:
        """Train the Decision Tree model and calculate validation accuracy."""
        if SKLEARN_AVAILABLE:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
            )
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            self.accuracy = float(accuracy_score(y_test, preds))
            # Refit on full dataset for production predictions
            self.model.fit(self.X, self.y)
        else:
            # 75/25 stratified split evaluation
            train_X, test_X, train_y, test_y = [], [], [], []
            for idx, (row, lbl) in enumerate(zip(self.X, self.y)):
                if idx % 4 == 0:
                    test_X.append(row)
                    test_y.append(lbl)
                else:
                    train_X.append(row)
                    train_y.append(lbl)
            self.model.fit(train_X, train_y)
            preds = self.model.predict(test_X)
            correct = sum(1 for p, t in zip(preds, test_y) if p == t)
            self.accuracy = correct / len(test_y) if test_y else 1.0
            self.model.fit(self.X, self.y)

    def predict_student(
        self,
        attendance: float,
        assignment_marks: float,
        internal_marks: float,
        study_hours: float,
    ) -> Dict[str, object]:
        """Predict performance class, probabilities, and diagnostic improvement suggestions."""
        features = [[attendance, assignment_marks, internal_marks, study_hours]]
        predicted_class = str(self.model.predict(features)[0])
        proba_list = self.model.predict_proba(features)[0]
        classes = list(self.model.classes_)
        probabilities = {cls: float(prob) for cls, prob in zip(classes, proba_list)}

        weak_areas, suggestions, action_plan = self.generate_improvement_suggestions(
            attendance=attendance,
            assignment_marks=assignment_marks,
            internal_marks=internal_marks,
            study_hours=study_hours,
            predicted_class=predicted_class,
        )

        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "weak_areas": weak_areas,
            "suggestions": suggestions,
            "action_plan": action_plan,
        }

    def generate_improvement_suggestions(
        self,
        attendance: float,
        assignment_marks: float,
        internal_marks: float,
        study_hours: float,
        predicted_class: str,
    ) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
        """
        Advanced Feature:
        Analyze each academic metric to detect areas needing improvement,
        severity level, and specific actionable recommendations.
        """
        weak_areas: List[Dict[str, object]] = []
        suggestions: List[str] = []
        action_plan: List[str] = []

        # 1. Attendance Check
        if attendance < 60:
            weak_areas.append({
                "area": "Attendance",
                "current": f"{attendance:.1f}%",
                "target": "75.0% - 88.0%",
                "severity": "Critical",
            })
            suggestions.append(
                f"[CRITICAL] Attendance is very low ({attendance:.1f}%). You risk missing core lecture concepts and falling below minimum exam eligibility (75%)."
            )
            action_plan.append(
                "Attend all upcoming lectures consistently and collect missed class notes from peers/faculty."
            )
        elif attendance < 75:
            weak_areas.append({
                "area": "Attendance",
                "current": f"{attendance:.1f}%",
                "target": "80.0%+",
                "severity": "Moderate",
            })
            suggestions.append(
                f"[MODERATE] Attendance ({attendance:.1f}%) is below the recommended 75% threshold."
            )
            action_plan.append(
                "Improve weekly class attendance to at least 80% to strengthen classroom comprehension."
            )
        elif attendance < 88 and predicted_class != "Excellent":
            weak_areas.append({
                "area": "Attendance",
                "current": f"{attendance:.1f}%",
                "target": "88.0%+",
                "severity": "Minor",
            })
            suggestions.append(
                f"[MINOR] Attendance ({attendance:.1f}%) is satisfactory, but increasing it to 88%+ correlates with 'Excellent' performance."
            )

        # 2. Assignment Marks Check
        if assignment_marks < 55:
            weak_areas.append({
                "area": "Assignment Marks",
                "current": f"{assignment_marks:.1f}/100",
                "target": "70.0/100+",
                "severity": "Critical",
            })
            suggestions.append(
                f"[CRITICAL] Assignment score ({assignment_marks:.1f}/100) indicates gaps in practical application and timely submission."
            )
            action_plan.append(
                "Start assignments early, review grading rubrics carefully, and consult teaching assistants before submission."
            )
        elif assignment_marks < 75:
            weak_areas.append({
                "area": "Assignment Marks",
                "current": f"{assignment_marks:.1f}/100",
                "target": "80.0/100+",
                "severity": "Moderate",
            })
            suggestions.append(
                f"[MODERATE] Assignment marks ({assignment_marks:.1f}/100) have room for improvement to boost continuous assessment credit."
            )
            action_plan.append(
                "Practice extra problem sets and revise feedback from previous assignments to reach 80+."
            )
        elif assignment_marks < 86 and predicted_class != "Excellent":
            weak_areas.append({
                "area": "Assignment Marks",
                "current": f"{assignment_marks:.1f}/100",
                "target": "86.0/100+",
                "severity": "Minor",
            })
            suggestions.append(
                f"[MINOR] Assignment marks ({assignment_marks:.1f}/100) are good; refining edge cases and presentation can push them above 86."
            )

        # 3. Internal Marks Check
        if internal_marks < 55:
            weak_areas.append({
                "area": "Internal Marks",
                "current": f"{internal_marks:.1f}/100",
                "target": "70.0/100+",
                "severity": "Critical",
            })
            suggestions.append(
                f"[CRITICAL] Internal exam score ({internal_marks:.1f}/100) is critically low and heavily impacts final performance."
            )
            action_plan.append(
                "Solve past internal test papers under timed conditions and focus on high-weightage syllabus units."
            )
        elif internal_marks < 75:
            weak_areas.append({
                "area": "Internal Marks",
                "current": f"{internal_marks:.1f}/100",
                "target": "82.0/100+",
                "severity": "Moderate",
            })
            suggestions.append(
                f"[MODERATE] Internal marks ({internal_marks:.1f}/100) are moderate; stronger conceptual revision is needed before internal assessments."
            )
            action_plan.append(
                "Conduct weekly self-quizzes and clarify doubts with instructors prior to internal exams."
            )
        elif internal_marks < 86 and predicted_class != "Excellent":
            weak_areas.append({
                "area": "Internal Marks",
                "current": f"{internal_marks:.1f}/100",
                "target": "86.0/100+",
                "severity": "Minor",
            })
            suggestions.append(
                f"[MINOR] Internal marks ({internal_marks:.1f}/100) are solid; targeted revision of analytical questions will help reach the 'Excellent' tier."
            )

        # 4. Study Hours Check
        if study_hours < 2.5:
            weak_areas.append({
                "area": "Study Hours",
                "current": f"{study_hours:.1f} hrs/day",
                "target": "4.5 - 6.0 hrs/day",
                "severity": "Critical",
            })
            suggestions.append(
                f"[CRITICAL] Daily self-study time ({study_hours:.1f} hrs/day) is insufficient for mastering coursework."
            )
            action_plan.append(
                f"Increase daily focused study time from {study_hours:.1f} hrs to at least 4.5 hrs/day using structured Pomodoro blocks (50m study / 10m break)."
            )
        elif study_hours < 5.0:
            weak_areas.append({
                "area": "Study Hours",
                "current": f"{study_hours:.1f} hrs/day",
                "target": "5.5 - 7.0 hrs/day",
                "severity": "Moderate",
            })
            suggestions.append(
                f"[MODERATE] Study duration ({study_hours:.1f} hrs/day) is average; adding 1.5–2 hours of active recall daily will significantly lift grades."
            )
            action_plan.append(
                f"Create a fixed daily timetable expanding self-study from {study_hours:.1f} hrs to 6.0 hrs/day."
            )
        elif study_hours < 6.8 and predicted_class != "Excellent":
            weak_areas.append({
                "area": "Study Hours",
                "current": f"{study_hours:.1f} hrs/day",
                "target": "7.0+ hrs/day",
                "severity": "Minor",
            })
            suggestions.append(
                f"[MINOR] Study hours ({study_hours:.1f} hrs/day) are good; optimizing study quality with spaced repetition can yield top-tier results."
            )

        # Cross-feature insight: High study hours but low internal marks
        if study_hours >= 5.5 and internal_marks < 65:
            suggestions.append(
                "[INSIGHT] You invest good study hours, but internal test marks remain low. Shift from passive reading to active problem-solving and mock tests."
            )

        # If no weak areas found (all metrics are top tier)
        if not suggestions:
            suggestions.append(
                "Outstanding academic profile across all parameters! Maintain your current consistency in attendance, coursework, and self-study."
            )
            action_plan.append(
                "Continue peer mentoring, explore advanced elective projects, and maintain your balanced study routine."
            )

        return weak_areas, suggestions, action_plan

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance weights learned by the Decision Tree."""
        return {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, self.model.feature_importances_)
        }

    def get_tree_rules(self) -> str:
        """Export human-readable Decision Tree rules."""
        if SKLEARN_AVAILABLE:
            return export_text(self.model, feature_names=FEATURE_NAMES)
        return self.model.export_rules(FEATURE_NAMES)


# ============================================================================
# CLI Display & Interactive Interface
# ============================================================================
def print_banner() -> None:
    print("=" * 72)
    print("       INTELLIGENT STUDENT PERFORMANCE PREDICTION SYSTEM (AI/ML)       ")
    print("         Algorithm: Supervised Classification - Decision Tree          ")
    print("=" * 72)


def display_model_summary(system: StudentPerformanceSystem) -> None:
    engine = "scikit-learn DecisionTreeClassifier" if SKLEARN_AVAILABLE else "CART DecisionTreeClassifier (Built-in)"
    print(f"\n[AI Model Status]")
    print(f"  Classifier Engine  : {engine}")
    print(f"  Training Samples   : {len(system.X)} records ({system.csv_path})")
    print(f"  Validation Accuracy: {system.accuracy * 100:.2f}%")
    print(f"  Target Classes     : Excellent | Good | Average | Poor")

    print("\n[Learned Feature Importances (Decision Tree)]")
    for feat, imp in system.get_feature_importances().items():
        bar = "#" * int(round(imp * 30))
        print(f"  - {feat:<18}: {imp * 100:5.1f}%  {bar}")


def display_prediction_report(
    student_name: str,
    student_id: str,
    attendance: float,
    assignment_marks: float,
    internal_marks: float,
    study_hours: float,
    result: Dict[str, object],
) -> None:
    predicted_class = str(result["predicted_class"])
    weak_areas: List[Dict[str, object]] = result["weak_areas"]  # type: ignore[assignment]
    suggestions: List[str] = result["suggestions"]  # type: ignore[assignment]
    action_plan: List[str] = result["action_plan"]  # type: ignore[assignment]

    badge_map = {
        "Excellent": "EXCELLENT (Top Tier Performance)",
        "Good": "GOOD (Above Average Performance)",
        "Average": "AVERAGE (Needs Moderate Improvement)",
        "Poor": "POOR (Requires Immediate Academic Intervention)",
    }

    print("\n" + "=" * 72)
    print("                     STUDENT PERFORMANCE REPORT                         ")
    print("=" * 72)
    print(f"  Student Name      : {student_name}")
    print(f"  Student ID / Roll : {student_id}")
    print("-" * 72)
    print("  INPUT ACADEMIC METRICS:")
    print(f"    1. Attendance       : {attendance:6.1f} %")
    print(f"    2. Assignment Marks : {assignment_marks:6.1f} / 100")
    print(f"    3. Internal Marks   : {internal_marks:6.1f} / 100")
    print(f"    4. Study Hours      : {study_hours:6.1f} hrs/day")
    print("-" * 72)
    print(f"  PREDICTED PERFORMANCE : {predicted_class.upper()}")
    print(f"  ASSESSMENT SUMMARY    : {badge_map.get(predicted_class, predicted_class)}")
    print("-" * 72)

    # Advanced Feature: Areas Where the Student Needs Improvement
    print("  AREAS NEEDING IMPROVEMENT (Diagnostic Analysis):")
    if not weak_areas:
        print("    None! All academic parameters meet or exceed 'Excellent' benchmarks.")
    else:
        print(f"    {'Parameter':<20} | {'Current':<14} | {'Recommended':<16} | {'Priority'}")
        print("    " + "-" * 64)
        for item in weak_areas:
            print(
                f"    {str(item['area']):<20} | {str(item['current']):<14} | "
                f"{str(item['target']):<16} | {str(item['severity'])}"
            )

    print("-" * 72)
    print("  PERSONALIZED SUGGESTIONS FOR IMPROVEMENT:")
    for idx, suggestion in enumerate(suggestions, start=1):
        print(f"    {idx}. {suggestion}")

    if action_plan:
        print("\n  RECOMMENDED ACTION PLAN:")
        for idx, step in enumerate(action_plan, start=1):
            print(f"    [{idx}] {step}")

    print("=" * 72 + "\n")


def prompt_float(prompt_text: str, min_val: float, max_val: float) -> float:
    """Prompt the user for a validated numeric value within [min_val, max_val]."""
    while True:
        raw = input(prompt_text).strip()
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  [!] Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  [!] Invalid input. Please enter a numeric value.")


def run_demo_cases(system: StudentPerformanceSystem) -> None:
    """Run sample predictions across all four performance tiers."""
    print("\n" + "=" * 72)
    print("        RUNNING SAMPLE PREDICTIONS ACROSS ALL PERFORMANCE TIERS         ")
    print("=" * 72)

    demo_students = [
        {
            "name": "Aarav Sharma",
            "id": "STU-101",
            "attendance": 94.0,
            "assignment_marks": 91.0,
            "internal_marks": 89.0,
            "study_hours": 7.5,
        },
        {
            "name": "Priya Nair",
            "id": "STU-102",
            "attendance": 81.0,
            "assignment_marks": 78.0,
            "internal_marks": 74.0,
            "study_hours": 5.0,
        },
        {
            "name": "Rohan Verma",
            "id": "STU-103",
            "attendance": 66.0,
            "assignment_marks": 61.0,
            "internal_marks": 58.0,
            "study_hours": 3.2,
        },
        {
            "name": "Kabir Patel",
            "id": "STU-104",
            "attendance": 48.0,
            "assignment_marks": 42.0,
            "internal_marks": 39.0,
            "study_hours": 1.5,
        },
    ]

    for s in demo_students:
        result = system.predict_student(
            attendance=s["attendance"],
            assignment_marks=s["assignment_marks"],
            internal_marks=s["internal_marks"],
            study_hours=s["study_hours"],
        )
        display_prediction_report(
            student_name=s["name"],
            student_id=s["id"],
            attendance=s["attendance"],
            assignment_marks=s["assignment_marks"],
            internal_marks=s["internal_marks"],
            study_hours=s["study_hours"],
            result=result,
        )


def interactive_cli(system: StudentPerformanceSystem) -> None:
    """Interactive menu loop for predicting student performance."""
    while True:
        print("\nChoose an option:")
        print("  1. Predict Performance for a New Student")
        print("  2. Run Demo Predictions (Excellent / Good / Average / Poor)")
        print("  3. View Learned Decision Tree Rules")
        print("  4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            print("\n--- Enter Student Details ---")
            name = input("  Student Name                  : ").strip() or "Anonymous Student"
            student_id = input("  Student Roll / ID             : ").strip() or "N/A"
            attendance = prompt_float("  Attendance (%) [0 - 100]      : ", 0.0, 100.0)
            assignment_marks = prompt_float("  Assignment Marks [0 - 100]    : ", 0.0, 100.0)
            internal_marks = prompt_float("  Internal Marks [0 - 100]      : ", 0.0, 100.0)
            study_hours = prompt_float("  Study Hours/Day [0 - 16]      : ", 0.0, 16.0)

            result = system.predict_student(
                attendance=attendance,
                assignment_marks=assignment_marks,
                internal_marks=internal_marks,
                study_hours=study_hours,
            )
            display_prediction_report(
                student_name=name,
                student_id=student_id,
                attendance=attendance,
                assignment_marks=assignment_marks,
                internal_marks=internal_marks,
                study_hours=study_hours,
                result=result,
            )
        elif choice == "2":
            run_demo_cases(system)
        elif choice == "3":
            print("\n[Decision Tree Classification Rules]")
            print(system.get_tree_rules())
        elif choice == "4":
            print("\nExiting Intelligent Student Performance Prediction System. Goodbye!")
            break
        else:
            print("  [!] Invalid selection. Please choose 1, 2, 3, or 4.")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "student_dataset.csv")

    print_banner()
    system = StudentPerformanceSystem(csv_path=csv_path)
    display_model_summary(system)

    if "--demo" in sys.argv:
        print("\n[Decision Tree Classification Rules]")
        print(system.get_tree_rules())
        run_demo_cases(system)
        return

    interactive_cli(system)


if __name__ == "__main__":
    main()
