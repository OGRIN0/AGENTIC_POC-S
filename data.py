import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataValidator:
    """A class for performing comprehensive data validation tests"""
    
    def __init__(self, data, target_column=None):
        """
        Initialize the validator with data
        
        Args:
            data (pd.DataFrame): Dataset to validate
            target_column (str): Name of the target/label column
        """
        self.data = data
        self.target_column = target_column
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column:
            if target_column in self.numeric_columns:
                self.numeric_columns.remove(target_column)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
    
    def check_missing_values(self, threshold=0.2):
        """
        Check for columns with missing values above threshold
        
        Args:
            threshold (float): Maximum acceptable proportion of missing values
            
        Returns:
            dict: Results of missing value analysis
        """
        missing_counts = self.data.isnull().sum()
        missing_percentage = missing_counts / len(self.data)
        
        columns_above_threshold = missing_percentage[missing_percentage > threshold].index.tolist()
        
        return {
            'missing_counts': missing_counts,
            'missing_percentage': missing_percentage,
            'columns_above_threshold': columns_above_threshold,
            'passed': len(columns_above_threshold) == 0
        }
    
    def check_data_balance(self, imbalance_threshold=0.2):
        """
        For classification problems, check if classes are balanced
        
        Args:
            imbalance_threshold (float): Maximum acceptable class imbalance ratio difference
            
        Returns:
            dict: Results of class balance analysis
        """
        if not self.target_column:
            return {'error': 'Target column not specified'}
            
        value_counts = self.data[self.target_column].value_counts()
        total = len(self.data)
        proportions = value_counts / total
        
        min_prop = proportions.min()
        max_prop = proportions.max()
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        return {
            'value_counts': value_counts,
            'proportions': proportions,
            'imbalance_ratio': imbalance_ratio,
            'passed': imbalance_ratio <= (1 + imbalance_threshold) / (1 - imbalance_threshold)
        }
    
    def check_feature_correlation(self, correlation_threshold=0.9):
        """
        Check for highly correlated features that might be redundant
        
        Args:
            correlation_threshold (float): Threshold above which features are considered highly correlated
            
        Returns:
            dict: Results of correlation analysis
        """
        if len(self.numeric_columns) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}
            
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        
        # Find pairs of features with correlation above threshold
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    high_correlation_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_correlation_pairs,
            'passed': len(high_correlation_pairs) == 0
        }
    
    def check_outliers(self, z_score_threshold=3.0):
        """
        Identify outliers in numeric features using z-score
        
        Args:
            z_score_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            dict: Results of outlier analysis
        """
        outliers_summary = {}
        
        for column in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.data[column].dropna()))
            outliers_mask = z_scores > z_score_threshold
            outliers_count = np.sum(outliers_mask)
            outliers_percentage = outliers_count / len(z_scores) * 100
            
            outliers_summary[column] = {
                'count': outliers_count,
                'percentage': outliers_percentage,
                'indexes': np.where(outliers_mask)[0].tolist()
            }
        
        return {
            'outliers_summary': outliers_summary,
            'columns_with_outliers': [col for col in outliers_summary if outliers_summary[col]['count'] > 0],
            'passed': all(outliers_summary[col]['percentage'] < 5 for col in outliers_summary)
        }
    
    def check_feature_distribution(self):
        """
        Analyze the distribution of numeric features for normality
        
        Returns:
            dict: Results of distribution analysis
        """
        distribution_results = {}
        
        for column in self.numeric_columns:
            if len(self.data[column].dropna()) < 8:  # Need at least 8 samples for shapiro test
                continue
                
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(self.data[column].dropna())
            
            # Skewness and kurtosis
            skewness = stats.skew(self.data[column].dropna())
            kurtosis = stats.kurtosis(self.data[column].dropna())
            
            distribution_results[column] = {
                'shapiro_stat': stat,
                'shapiro_p_value': p_value,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': p_value > 0.05
            }
        
        return {
            'distribution_results': distribution_results,
            'normal_features': [col for col in distribution_results if distribution_results[col]['is_normal']],
            'non_normal_features': [col for col in distribution_results if not distribution_results[col]['is_normal']]
        }
    
    def check_data_leakage(self, time_column=None):
        """
        Check for potential data leakage issues
        
        Args:
            time_column (str): Column name containing timestamps (if time series data)
            
        Returns:
            dict: Results of data leakage analysis
        """
        leakage_issues = []
        
        # Check for duplicate rows
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            leakage_issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for target-correlated features that might indicate leakage
        if self.target_column:
            leakage_concerns = []
            
            # For numeric target, check correlation
            if self.target_column in self.data.select_dtypes(include=[np.number]).columns:
                for col in self.numeric_columns:
                    correlation = self.data[[col, self.target_column]].corr().iloc[0, 1]
                    if abs(correlation) > 0.95:  # Extremely high correlation can indicate leakage
                        leakage_concerns.append(f"Feature '{col}' has {correlation:.2f} correlation with target")
            
            # For categorical target, check predictive power
            else:
                from sklearn.metrics import mutual_info_score
                
                for col in self.numeric_columns:
                    mi = mutual_info_score(self.data[self.target_column], self.data[col])
                    if mi > 0.9:  # Very high mutual information can indicate leakage
                        leakage_concerns.append(f"Feature '{col}' has high mutual information ({mi:.2f}) with target")
            
            if leakage_concerns:
                leakage_issues.extend(leakage_concerns)
        
        # For time series, check if data is properly ordered
        if time_column and time_column in self.data.columns:
            if not self.data[time_column].equals(self.data[time_column].sort_values()):
                leakage_issues.append("Time series data is not sorted chronologically")
        
        return {
            'leakage_issues': leakage_issues,
            'passed': len(leakage_issues) == 0
        }
    
    def visualize_distributions(self, figsize=(15, 10)):
        """
        Create distribution plots for numeric features
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with distribution plots
        """
        n_cols = 3
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, column in enumerate(self.numeric_columns):
            if i < len(axes):
                sns.histplot(self.data[column].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {column}')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        return fig
    
    def run_all_tests(self):
        """
        Run all data validation tests and return comprehensive results
        
        Returns:
            dict: Results of all tests
        """
        results = {
            'missing_values': self.check_missing_values(),
            'feature_correlation': self.check_feature_correlation(),
            'outliers': self.check_outliers(),
            'feature_distribution': self.check_feature_distribution(),
            'data_leakage': self.check_data_leakage()
        }
        
        if self.target_column:
            results['data_balance'] = self.check_data_balance()
        
        # Overall test result
        all_passed = all(
            test.get('passed', True) 
            for test in results.values() 
            if isinstance(test, dict) and 'passed' in test
        )
        
        results['all_tests_passed'] = all_passed
        
        return results

# Example usage with Iris dataset
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

# Create validator
validator = DataValidator(iris_df, target_column='target')

# Run all tests
validation_results = validator.run_all_tests()

# Print summary
print("Data Validation Summary:")
print(f"All tests passed: {validation_results['all_tests_passed']}")

for test_name, test_result in validation_results.items():
    if isinstance(test_result, dict) and 'passed' in test_result:
        status = "✓ PASSED" if test_result['passed'] else "✗ FAILED"
        print(f"{test_name}: {status}")

# Visualize distributions
validator.visualize_distributions()
plt.show()


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, mean_absolute_percentage_error,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

class ModelTester:
    """A class for comprehensive testing and evaluation of ML models"""
    
    def __init__(self, model, is_classifier=True):
        """
        Initialize the model tester
        
        Args:
            model: Trained ML model with predict and predict_proba methods
            is_classifier (bool): Whether this is a classification model
        """
        self.model = model
        self.is_classifier = is_classifier
        self.performance_results = {}
    
    def evaluate_classifier(self, X_test, y_test, class_names=None):
        """
        Evaluate a classification model with multiple metrics
        
        Args:
            X_test: Test features
            y_test: True labels
            class_names (list): Names for classes
            
        Returns:
            dict: Classification performance metrics
        """
        if not self.is_classifier:
            return {'error': 'Model is not a classifier'}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get probabilities if available
        has_proba = hasattr(self.model, "predict_proba") and callable(getattr(self.model, "predict_proba"))
        
        if has_proba:
            y_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary vs multiclass cases
        if len(np.unique(y_test)) == 2:  # Binary classification
            average = 'binary'
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if has_proba:
                # For binary classification, we need the probability of class 1
                if y_proba.shape[1] >= 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba)
        else:  # Multiclass
            average = 'weighted'
            precision = precision_score(y_test, y_pred, average=average)
            recall = recall_score(y_test, y_pred, average=average)
            f1 = f1_score(y_test, y_pred, average=average)
            
            if has_proba:
                try:
                    # For multiclass, we use OvR approach
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except:
                    roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Detailed classification report
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_test)))]
            
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'average_method': average
        }
        
        if has_proba:
            results['roc_auc'] = roc_auc
        
        self.performance_results['classification'] = results
        return results
    
    def evaluate_regressor(self, X_test, y_test):
        """
        Evaluate a regression model with multiple metrics
        
        Args:
            X_test: Test features
            y_test: True values
            
        Returns:
            dict: Regression performance metrics
        """
        if self.is_classifier:
            return {'error': 'Model is not a regressor'}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE only if y_test doesn't contain zeros
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        except:
            mape = None
        
        # Residuals analysis
        residuals = y_test - y_pred
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'residuals': residuals,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals)
        }
        
        self.performance_results['regression'] = results
        return results
    
    def test_calibration(self, X_test, y_test, n_bins=10):
        """
        Test if a classifier's probabilities are well-calibrated
        
        Args:
            X_test: Test features
            y_test: True labels
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            dict: Calibration test results
        """
        if not self.is_classifier:
            return {'error': 'Model is not a classifier'}
            
        if not hasattr(self.model, "predict_proba"):
            return {'error': 'Model does not support probability predictions'}
        
        # Handle binary vs multiclass
        if len(np.unique(y_test)) == 2:  # Binary
            y_proba = self.model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
            
            results = {
                'fraction_of_positives': prob_true,
                'mean_predicted_value': prob_pred,
                'n_bins': n_bins
            }
            
            self.performance_results['calibration'] = results
            return results
        else:
            # For multiclass, calibration curves are more complex
            # We would typically do one-vs-rest calibration curves
            return {'error': 'Multiclass calibration not implemented'}
    
    def test_learning_curve(self, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Generate learning curves to assess model performance vs training size
        
        Args:
            X: All features
            y: All labels/targets
            cv (int): Number of cross-validation folds
            train_sizes (array): Relative or absolute training set sizes
            
        Returns:
            dict: Learning curve data
        """
        # Define scoring metric based on classifier/regressor
        if self.is_classifier:
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
        
        # Generate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=-1,
            train_sizes=train_sizes, scoring=scoring
        )
        
        # Calculate mean and std
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # For regression, convert negative MSE to positive
        if not self.is_classifier:
            train_scores_mean = -train_scores_mean
            test_scores_mean = -test_scores_mean
        
        results = {
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores_mean,
            'train_scores_std': train_scores_std,
            'test_scores_mean': test_scores_mean,
            'test_scores_std': test_scores_std,
            'metric': 'accuracy' if self.is_classifier else 'mse'
        }
        
        self.performance_results['learning_curve'] = results
        return results
    
    def test_error_distribution(self, X_test, y_test):
        """
        Analyze the distribution of prediction errors
        
        Args:
            X_test: Test features
            y_test: True labels/values
            
        Returns:
            dict: Error distribution analysis
        """
        y_pred = self.model.predict(X_test)
        
        if self.is_classifier:
            # For classifiers, analyze which classes are misclassified most often
            correct_mask = y_pred == y_test
            incorrect_indices = np.where(~correct_mask)[0]
            
            class_error_counts = {}
            for idx in incorrect_indices:
                true_class = y_test[idx]
                pred_class = y_pred[idx]
                
                key = f"{true_class}->{pred_class}"
                class_error_counts[key] = class_error_counts.get(key, 0) + 1
            
            results = {
                'correct_predictions': np.sum(correct_mask),
                'incorrect_predictions': len(incorrect_indices),
                'accuracy': np.mean(correct_mask),
                'class_error_counts': class_error_counts,
                'most_confused_classes': sorted(
                    class_error_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5] if class_error_counts else []
            }
        else:
            # For regressors, analyze the distribution of errors
            errors = y_test - y_pred
            abs_errors = np.abs(errors)
            
            results = {
                'errors_mean': np.mean(errors),
                'errors_std': np.std(errors),
                'errors_min': np.min(errors),
                'errors_max': np.max(errors),
                'abs_errors_mean': np.mean(abs_errors),
                'abs_errors_median': np.median(abs_errors),
                'abs_errors_90th_percentile': np.percentile(abs_errors, 90),
                'errors_normality': stats.shapiro(errors)
            }
        
        self.performance_results['error_distribution'] = results
        return results
    
    def visualize_results(self, X_test=None, y_test=None):
        """
        Visualize model performance results
        
        Args:
            X_test: Test features (needed for some visualizations)
            y_test: True labels/values (needed for some visualizations)
            
        Returns:
            matplotlib.figure.Figure: Figure with performance visualizations
        """
        if self.is_classifier:
            return self._visualize_classifier_results(X_test, y_test)
        else:
            return self._visualize_regressor_results(X_test, y_test)
    
    def _visualize_classifier_results(self, X_test, y_test):
        """Helper method to visualize classifier results"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Confusion Matrix
        if 'classification' in self.performance_results:
            cm = self.performance_results['classification']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
            axs[0, 0].set_title('Confusion Matrix')
            axs[0, 0].set_xlabel('Predicted Label')
            axs[0, 0].set_ylabel('True Label')
        
        # Plot 2: ROC Curve (for binary classification)
        if 'classification' in self.performance_results and X_test is not None and y_test is not None:
            if len(np.unique(y_test)) == 2 and hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]
                
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                              label=f'ROC curve (area = {roc_auc:.2f})')
                axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axs[0, 1].set_xlim([0.0, 1.0])
                axs[0, 1].set_ylim([0.0, 1.05])
                axs[0, 1].set_xlabel('False Positive Rate')
                axs[0, 1].set_ylabel('True Positive Rate')
                axs[0, 1].set_title('ROC Curve')
                axs[0, 1].legend(loc="lower right")
            else:
                axs[0, 1].text(0.5, 0.5, 'ROC curve only available for binary classification',
                              horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Calibration Curve
        if 'calibration' in self.performance_results:
            prob_true = self.performance_results['calibration']['fraction_of_positives']
            prob_pred = self.performance_results['calibration']['mean_predicted_value']
            
            axs[1, 0].plot(prob_pred, prob_true, marker='o', linewidth=1)
            axs[1, 0].plot([0, 1], [0, 1], 'k--')
            axs[1, 0].set_xlabel('Mean Predicted Probability')
            axs[1, 0].set_ylabel('Fraction of Positives')
            axs[1, 0].set_title('Calibration Curve')
            
        # Plot 4: Learning Curve
        if 'learning_curve' in self.performance_results:
            lc = self.performance_results['learning_curve']
            axs[1, 1].plot(lc['train_sizes'], lc['train_scores_mean'], 'o-', color='r', label='Training score')
            axs[1, 1].plot(lc['train_sizes'], lc['test_scores_mean'], 'o-', color='g', label='Cross-validation score')
            axs[1, 1].fill_between(lc['train_sizes'], 
                                  lc['train_scores_mean'] - lc['train_scores_std'],
                                  lc['train_scores_mean'] + lc['train_scores_std'], 
                                  alpha=0.1, color='r')
            axs[1, 1].fill_between(lc['train_sizes'], 
                                  lc['test_scores_mean'] - lc['test_scores_std'],
                                  lc['test_scores_mean'] + lc['test_scores_std'], 
                                  alpha=0.1, color='g')
            axs[1, 1].set_xlabel('Training Examples')
            axs[1, 1].set_ylabel('Score')
            axs[1, 1].set_title('Learning Curve')
            axs[1, 1].legend(loc='best')
            
        plt.tight_layout()
        return fig
    
    def _visualize_regressor_results(self, X_test, y_test):
        """Helper method to visualize regressor results"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Predicted vs Actual
        if X_test is not None and y_test is not None:
            y_pred = self.model.predict(X_test)
            axs[0, 0].scatter(y_test, y_pred, alpha=0.3)
            axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            axs[0, 0].set_xlabel('Actual')
            axs[0, 0].set_ylabel('Predicted')
            axs[0, 0].set_title('Predicted vs Actual')
            
        # Plot 2: Residuals
        if 'regression' in self.performance_results and X_test is not None and y_test is not None:
            residuals = self.performance_results['regression']['residuals']
            axs[0, 1].scatter(y_pred, residuals, alpha=0.3)
            axs[0, 1].axhline(y=0, color='k', linestyle='--', lw=2)
            axs[0, 1].set_xlabel('Predicted')
            axs[0, 1].set_ylabel('Residuals')
            axs[0, 1].set_title('Residuals vs Predicted')
            
        # Plot 3: Residual Distribution
        if 'regression' in self.performance_results:
            residuals = self.performance_results['regression']['residuals']
            sns.histplot(residuals, kde=True, ax=axs[1, 0])
            axs[