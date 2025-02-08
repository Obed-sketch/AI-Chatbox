# %% [8] Loan Calculator Module
class LoanCalculator:
    @staticmethod
    def calculate_monthly_payment(principal, annual_rate, years):
        """Calculate monthly payment using annuity formula"""
        monthly_rate = annual_rate / 12 / 100
        periods = years * 12
        payment = principal * (monthly_rate * (1 + monthly_rate)**periods) / \
                 ((1 + monthly_rate)**periods - 1)
        return round(payment, 2)

    @staticmethod
    def calculate_amortization(principal, annual_rate, years):
        """Generate full amortization schedule"""
        monthly_payment = LoanCalculator.calculate_monthly_payment(
            principal, annual_rate, years
        )
        balance = principal
        monthly_rate = annual_rate / 12 / 100
        schedule = []
        
        for month in range(1, years * 12 + 1):
            interest = balance * monthly_rate
            principal_payment = monthly_payment - interest
            balance -= principal_payment
            schedule.append({
                "Month": month,
                "Payment": monthly_payment,
                "Principal": round(principal_payment, 2),
                "Interest": round(interest, 2),
                "Remaining Balance": abs(round(balance, 2))
            })
        return pd.DataFrame(schedule)

    @staticmethod
    def calculate_debt_to_income(monthly_debt, monthly_income):
        """Calculate DTI ratio"""
        return round((monthly_debt / monthly_income) * 100, 2)

# %% [9] Fraud Detection Module
class FraudDetector:
    def __init__(self):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=0.01)
        self.transaction_history = pd.DataFrame()
        self.fraud_rules = {
            'max_amount': 10000,
            'velocity_window': 5,  # minutes
            'velocity_count': 3
        }
    
    def add_transaction(self, transaction):
        """Add transaction to history"""
        self.transaction_history = self.transaction_history.append(
            transaction, ignore_index=True
        )
    
    def detect_fraud(self, new_transaction):
        """Analyze transaction for fraud patterns"""
        # Rule-based checks
        amount_flag = new_transaction['amount'] > self.fraud_rules['max_amount']
        
        time_window = pd.Timestamp.now() - pd.Timedelta(
            minutes=self.fraud_rules['velocity_window']
        )
        velocity_flag = len(self.transaction_history[
            (self.transaction_history['account'] == new_transaction['account']) &
            (self.transaction_history['timestamp'] >= time_window)
        ]) >= self.fraud_rules['velocity_count']
        
        # Machine learning check
        features = self._extract_features(new_transaction)
        ml_flag = self.model.predict([features])[0] == -1
        
        return {
            'amount_flag': amount_flag,
            'velocity_flag': velocity_flag,
            'ml_flag': ml_flag,
            'fraud_probability': self._calculate_probability(
                amount_flag, velocity_flag, ml_flag
            )
        }
    
    def _extract_features(self, transaction):
        """Create ML features from transaction data"""
        return [
            transaction['amount'],
            len(self.transaction_history[
                self.transaction_history['account'] == transaction['account']
            ]),
            pd.Timestamp(transaction['timestamp']).hour
        ]
    
    def _calculate_probability(self, *flags):
        """Calculate combined fraud probability"""
        weights = [0.4, 0.3, 0.3]
        return sum(weight * flag for weight, flag in zip(weights, flags))

# %% [10] Updated BankingAI Class
class BankingAI:
    def __init__(self):
        # Previous initialization
        self.loan_calculator = LoanCalculator()
        self.fraud_detector = FraudDetector()
    
    def analyze_query(self, text):
        # Existing analysis
        if "loan" in response.lower():
            return self._handle_loan_queries(text)
        return response
    
    def _handle_loan_queries(self, text):
        """Process loan-related queries"""
        doc = self.nlp(text)
        amounts = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
        rates = [ent.text for ent in doc.ents if ent.label_ == "PERCENT"]
        
        if amounts and rates:
            principal = float(amounts[0].replace('$', ''))
            rate = float(rates[0].replace('%', ''))
            payment = self.loan_calculator.calculate_monthly_payment(
                principal, rate, 30
            )
            return f"Estimated monthly payment: ${payment}"
        return "Please provide loan amount and interest rate"

# %% [11] New Flask Routes
@app.route('/loan', methods=['POST'])
@requires_auth
def handle_loan():
    data = request.json
    calculation_type = data.get('type')
    
    try:
        if calculation_type == 'monthly_payment':
            result = ai_system.loan_calculator.calculate_monthly_payment(
                data['principal'],
                data['annual_rate'],
                data['years']
            )
        elif calculation_type == 'amortization':
            result = ai_system.loan_calculator.calculate_amortization(
                data['principal'],
                data['annual_rate'],
                data['years']
            ).to_dict()
        elif calculation_type == 'dti':
            result = ai_system.loan_calculator.calculate_debt_to_income(
                data['monthly_debt'],
                data['monthly_income']
            )
        else:
            return jsonify({"error": "Invalid calculation type"}), 400
            
        return jsonify({
            "result": result,
            "calculation_type": calculation_type
        })
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400

@app.route('/fraud', methods=['POST'])
@requires_auth
def detect_fraud():
    transaction = request.json
    try:
        result = ai_system.fraud_detector.detect_fraud(transaction)
        ai_system.fraud_detector.add_transaction(transaction)
        return jsonify({
            "fraud_indicators": result,
            "transaction_id": hashlib.sha256(
                str(transaction).encode()
            ).hexdigest()[:10]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# %% [12] Enhanced Backtesting
class EnhancedBacktester(Backtester):
    def __init__(self, ai_model):
        super().__init__(ai_model)
        self.fraud_cases = [
            {'amount': 15000, 'account': 'A123', 'timestamp': '2023-01-01 12:00'},
            {'amount': 500, 'account': 'B456', 'timestamp': '2023-01-01 12:04'},
            {'amount': 500, 'account': 'B456', 'timestamp': '2023-01-01 12:05'},
            {'amount': 500, 'account': 'B456', 'timestamp': '2023-01-01 12:06'},
        ]
    
    def test_fraud_detection(self):
        results = []
        for case in self.fraud_cases:
            detection = self.ai.fraud_detector.detect_fraud(case)
            results.append({
                "transaction": case,
                "fraud_probability": detection['fraud_probability'],
                "flags": detection
            })
        return pd.DataFrame(results)

# %% [13] Updated Deployment
if __name__ == "__main__":
    # Initialize with sample fraud data
    ai_system.fraud_detector.model.fit(
        np.random.rand(100, 3)  # Sample training data
    )
    
    # Run enhanced backtests
    enhanced_backtester = EnhancedBacktester(ai_system)
    fraud_results = enhanced_backtester.test_fraud_detection()
    fraud_results.to_excel("fraud_test_results.xlsx", index=False)
    
    # Start server
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
