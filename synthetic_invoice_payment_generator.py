import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from uuid import uuid4
from typing import Dict, List, Tuple, Optional
from faker import Faker
import json
import os
import logging
from enum import Enum
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from data_model import Fattura, Transazione
from dataclasses import dataclass, asdict
from dotenv import load_dotenv


class EnvironmentConfigError(Exception):
    """Raised when environment variables cannot be loaded or are missing"""
    pass


# Move environment loading to main() function for better error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchType(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    RELATED = "related"
    UNRELATED = "unrelated"

class QualityLevel(Enum):
    PERFECT = "perfect"
    FUZZY = "fuzzy"
    NOISY = "noisy"

class TimingPattern(Enum):
    STANDARD = "standard"  # 0-90 days
    DELAYED = "delayed"    # >90 days
    EARLY = "early"       # before invoice
    SAME_DAY = "same_day"  # same day

class AmountPattern(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    EXCESS = "excess"
    DISCOUNT = "discount"
    PENALTY = "penalty"

@dataclass
class GroundTruth:
    fattura_id: str
    pagamento_id: str
    match_type: str
    confidence: float
    amount_covered: float
    notes: str


class AITextGenerator:
    def __init__(self, azure_endpoint: str, api_version: str = "2024-08-01-preview", model: str = "gpt-4o", temperature: float = 0.7):
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            openai_api_version=api_version,
            deployment_name=model,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=temperature
        )
        self.llm_fattura = self.llm.with_structured_output(Fattura)
        self.llm_trans = self.llm.with_structured_output(Transazione)
        
    def generate_causale(self, prestatore: str, numero_fattura: str, importo: float, 
                        scenario_type: str, quality_level: QualityLevel) -> str:
        """Generate realistic payment causale"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Genera una stringa di testo che emuli realisticamente il dettaglio, la causale e la controparte di una transazione con i seguenti attributi:"),
            ("human", "{attributi_transazione}")
            ])

        attributi_transazione = f"""
        - id: {uuid4()}
        - Importo: €{importo:.2f}
        - Scenario: {scenario_type}
        """
        chain = prompt_template | self.llm_trans
        try:
            response : Transazione = chain.invoke({"attributi_transazione": attributi_transazione})
            return response.causale
        except Exception as e:
            logger.error(f"Error generating causale: {e}")
            return f"Pagamento fattura {numero_fattura} - {prestatore}"
    
    def generate_invoice_data(self, settore: str, prestatore: str, 
                             importo: float, tipo_servizio: str) -> Tuple[str, str]:
        """Generate realistic invoice description and committente"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Genera una fattura realistica emessa in Italia con descrizione dettagliata e committente appropriato per questi attributi:"),
            ("human", "{attributi_fattura}")
            ])
        attributi_fattura = f"""
        - id: {uuid4()}
        - Settore: {settore}
        - Prestatore: {prestatore}
        - Importo: €{importo:.2f}
        - Tipo servizio: {tipo_servizio}
        """
        chain = prompt_template | self.llm_fattura
        try:
            response : Fattura = chain.invoke({"attributi_fattura": attributi_fattura})
            return response.descrizione, response.committente
        except Exception as e:
            logger.error(f"Error generating invoice data: {e}")
            return f"Servizi {tipo_servizio} per {prestatore}", "AESON SRL"

    def generate_transaction_data(self, prestatore: str, importo: float, numero_fattura: str) -> Tuple[str, str, str]:
        """Generate realistic transaction dettaglio, causale and controparte"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Genera una transazione bancaria italiana realistica con dettaglio, causale e controparte per questi attributi:"),
            ("human", "{attributi_transazione}")
            ])
        attributi_transazione = f"""
        - id: {uuid4()}
        - Beneficiario: {prestatore}
        - Importo: €{importo:.2f}
        - Numero fattura: {numero_fattura}
        """
        
        chain = prompt_template | self.llm_trans

        try:
            response : Transazione = chain.invoke({"attributi_transazione": attributi_transazione})
            return response.dettaglio, response.causale, response.controparte
        except Exception as e:
            logger.error(f"Error generating transaction data: {e}")
            return f"Bonifico a favore di: {prestatore}", f"Pagamento fattura {numero_fattura}", prestatore


class SyntheticDataGenerator:
    def __init__(self, config: Dict, azure_endpoint: str):
        self.config = config
        self.fake = Faker('it_IT')
        self.ai_generator = AITextGenerator(azure_endpoint)
        self.companies = []
        self.sectors = [
            "Consulenza IT", "Servizi Legali", "Marketing", "Contabilità", 
            "Ingegneria", "Architettura", "Formazione", "Logistica"
        ]
        self.service_types = {
            "Consulenza IT": ["Sviluppo software", "Manutenzione sistemi", "Consulenza tecnica"],
            "Servizi Legali": ["Consulenza legale", "Assistenza contrattuale", "Rappresentanza"],
            "Marketing": ["Campagne pubblicitarie", "Social media management", "Branding"],
            "Contabilità": ["Tenuta contabilità", "Consulenza fiscale", "Bilanci"],
            "Ingegneria": ["Progettazione", "Collaudi", "Consulenza tecnica"],
            "Architettura": ["Progettazione architettonica", "Direzione lavori", "Pratiche edilizie"],
            "Formazione": ["Corsi di formazione", "Seminari", "Coaching"],
            "Logistica": ["Trasporti", "Magazzinaggio", "Distribuzione"]
        }
        
    def generate_companies(self, n: int) -> List[Dict]:
        """Generate realistic company data"""
        companies = []
        for _ in range(n):
            sector = random.choice(self.sectors)
            company = {
                'cliente_id': str(uuid.uuid4()),
                'nome': self.fake.company(),
                'piva': self.fake.vat_id(),
                'settore': sector,
                'iban': self.fake.iban()
            }
            companies.append(company)
        self.companies = companies
        return companies
    
    def _generate_amount_distribution(self, base_amount: float, pattern: AmountPattern) -> float:
        """Generate amount based on pattern"""
        if pattern == AmountPattern.EXACT:
            return base_amount
        elif pattern == AmountPattern.PARTIAL:
            return base_amount * random.uniform(0.3, 0.9)
        elif pattern == AmountPattern.EXCESS:
            return base_amount * random.uniform(1.01, 1.2)
        elif pattern == AmountPattern.DISCOUNT:
            return base_amount * random.uniform(0.85, 0.98)
        elif pattern == AmountPattern.PENALTY:
            return base_amount * random.uniform(1.02, 1.15)
        return base_amount
    
    def _generate_timing_pattern(self, invoice_date: datetime, pattern: TimingPattern) -> datetime:
        """Generate payment date based on timing pattern"""
        if pattern == TimingPattern.STANDARD:
            days_offset = random.randint(0, 90)
        elif pattern == TimingPattern.DELAYED:
            days_offset = random.randint(91, 365)
        elif pattern == TimingPattern.EARLY:
            days_offset = random.randint(-30, -1)
        elif pattern == TimingPattern.SAME_DAY:
            days_offset = 0
        else:
            days_offset = random.randint(0, 90)
            
        return invoice_date + timedelta(days=days_offset)
    
    def _apply_noise_to_text(self, text: str, quality: QualityLevel) -> str:
        """Apply noise to text based on quality level"""
        if quality == QualityLevel.PERFECT:
            return text
        elif quality == QualityLevel.FUZZY:
            # Small variations
            if random.random() < 0.3:
                text = text.replace(' ', '  ')  # Double spaces
            if random.random() < 0.2:
                text = text.upper() if random.random() < 0.5 else text.lower()
            return text
        elif quality == QualityLevel.NOISY:
            # Typos and formatting issues
            chars = list(text)
            for i in range(len(chars)):
                if random.random() < 0.05:  # 5% chance of typo
                    if chars[i].isalpha():
                        chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            return ''.join(chars)
        return text
    
    def _generate_scadenza_date(self, data_emissione: datetime) -> datetime:
        """Generate a realistic scadenza date (30-90 days after emission)"""
        days_to_add = random.choice([30, 60, 90])  # Common payment terms
        return data_emissione + timedelta(days=days_to_add)
    
    def generate_invoice(self, company: Dict, amount_range: Tuple[float, float] = (100, 10000)) -> Fattura:
        """Generate a single invoice using Fattura model"""
        # Generate dates
        data_emissione = self.fake.date_between(start_date='-1y', end_date='today')
        data_scadenza = self._generate_scadenza_date(data_emissione)
        
        # Generate amounts
        importo = random.uniform(*amount_range)
        
        # Generate description and committente using AI in single call
        tipo_servizio = random.choice(self.service_types[company['settore']])
        descrizione, committente = self.ai_generator.generate_invoice_data(
            company['settore'], company['nome'], importo, tipo_servizio
        )
        
        return Fattura(
            data_emissione=data_emissione,
            data_scadenza=data_scadenza,
            descrizione=descrizione,
            importo=importo,
            prestatore=company['nome'],
            committente=committente
        )
    
    def generate_payment(self, fattura: Fattura, company: Dict, 
                        amount_pattern: AmountPattern = AmountPattern.EXACT,
                        timing_pattern: TimingPattern = TimingPattern.STANDARD,
                        quality_level: QualityLevel = QualityLevel.PERFECT) -> Transazione:
        """Generate a payment using Transazione model"""
        
        # Generate payment amount based on pattern
        payment_amount = self._generate_amount_distribution(fattura.importo, amount_pattern)
        
        # Generate payment date based on timing pattern
        payment_date = self._generate_timing_pattern(fattura.data_emissione, timing_pattern)
        
        # Generate numero fattura for reference
        numero_fattura = f"FT{fattura.data_emissione.year}{random.randint(1000, 9999)}"
        
        # Generate dettaglio, causale and controparte using AI in single call
        dettaglio, causale, controparte = self.ai_generator.generate_transaction_data(
            fattura.prestatore, payment_amount, numero_fattura
        )
        
        # Apply noise to causale based on quality level
        causale = self._apply_noise_to_text(causale, quality_level)
        
        return Transazione(
            data=payment_date,
            dettaglio=dettaglio,
            importo=payment_amount,
            tipologia_movimento="pagamento",
            controparte=controparte,
            causale=causale
        )
    
    def generate_scenario_1_1_perfect(self, n_pairs: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate 1:1 perfect match scenario"""
        fatture = []
        transazioni = []
        ground_truth = []
        
        companies = self.generate_companies(n_pairs)
        
        for company in companies:
            # Generate invoice
            fattura = self.generate_invoice(company)
            fatture.append(fattura)
            
            # Generate matching payment
            transazione = self.generate_payment(
                fattura, company, 
                AmountPattern.EXACT, 
                TimingPattern.STANDARD, 
                QualityLevel.PERFECT
            )
            transazioni.append(transazione)
            
            # Create ground truth
            gt = GroundTruth(
                fattura_id=str(fattura.id),
                pagamento_id=str(transazione.id),
                match_type=MatchType.EXACT.value,
                confidence=1.0,
                amount_covered=fattura.importo,
                notes="Perfect 1:1 match"
            )
            ground_truth.append(gt)
            
        return fatture, transazioni, ground_truth
    
    def generate_scenario_1_n_installments(self, n_invoices: int, installments_per_invoice: int = 3) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate 1:N installment payments scenario"""
        fatture = []
        transazioni = []
        ground_truth = []
        
        companies = self.generate_companies(n_invoices)
        
        for company in companies:
            # Generate invoice
            fattura = self.generate_invoice(company, (1000, 10000))  # Higher amounts for installments
            fatture.append(fattura)
            
            # Generate installment payments
            remaining_amount = fattura.importo
            for i in range(installments_per_invoice):
                if i == installments_per_invoice - 1:
                    # Last installment gets remaining amount
                    installment_amount = remaining_amount
                else:
                    # Random installment amount
                    installment_amount = remaining_amount * random.uniform(0.2, 0.4)
                    remaining_amount -= installment_amount
                
                # Create a temporary fattura with installment amount for payment generation
                temp_fattura = Fattura(
                    data_emissione=fattura.data_emissione,
                    data_scadenza=fattura.data_scadenza,
                    descrizione=fattura.descrizione,
                    importo=installment_amount,
                    prestatore=fattura.prestatore,
                    committente=fattura.committente
                )
                
                transazione = self.generate_payment(
                    temp_fattura, company,
                    AmountPattern.EXACT,
                    TimingPattern.STANDARD,
                    QualityLevel.PERFECT
                )
                transazioni.append(transazione)
                
                # Create ground truth
                gt = GroundTruth(
                    fattura_id=str(fattura.id),
                    pagamento_id=str(transazione.id),
                    match_type=MatchType.PARTIAL.value,
                    confidence=0.9,
                    amount_covered=installment_amount,
                    notes=f"Installment {i+1}/{installments_per_invoice}"
                )
                ground_truth.append(gt)
                
        return fatture, transazioni, ground_truth
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Generate complete synthetic dataset"""
        all_fatture = []
        all_transazioni = []
        all_ground_truth = []
        
        # Generate different scenarios based on config
        scenarios_config = self.config.get('scenarios', {})
        
        # 1:1 Perfect Match
        if scenarios_config.get('perfect_1_1', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_1_1_perfect(
                scenarios_config['perfect_1_1']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)
        
        # 1:N Installments
        if scenarios_config.get('installments_1_n', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_1_n_installments(
                scenarios_config['installments_1_n']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)
        
        # Convert to DataFrames
        fatture_df = pd.DataFrame([fattura.model_dump() for fattura in all_fatture])
        transazioni_df = pd.DataFrame([transazione.model_dump() for transazione in all_transazioni])
        ground_truth_df = pd.DataFrame([asdict(gt) for gt in all_ground_truth])
        
        # Generate metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_fatture': len(all_fatture),
            'total_transazioni': len(all_transazioni),
            'total_matches': len(all_ground_truth),
            'scenarios': scenarios_config,
            'config': self.config
        }
        
        return fatture_df, transazioni_df, ground_truth_df, metadata
    
    def export_dataset(self, dataset: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict], 
                      output_dir: str = "output"):
        """Export dataset to multiple formats"""
        fatture_df, transazioni_df, ground_truth_df, metadata = dataset
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        fatture_df.to_csv(f"{output_dir}/invoices.csv", index=False)
        transazioni_df.to_csv(f"{output_dir}/payments.csv", index=False)
        ground_truth_df.to_csv(f"{output_dir}/ground_truth.csv", index=False)
        
        # Export metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Dataset exported to {output_dir}")


DEFAULT_CONFIG = {
    'scenarios': {
        'perfect_1_1': 5,
        'installments_1_n': 3,
    },
    'quality_distribution': {
        'perfect': 0.4,
        'fuzzy': 0.5,
        'noisy': 0.1
    },
    'timing_distribution': {
        'standard': 0.6,
        'delayed': 0.2,
        'early': 0.1,
        'same_day': 0.1
    }
}

def main():
    try:
        # Load environment variables
        success = load_dotenv()
        if not success:
            raise EnvironmentConfigError('Cannot load .env file. Please ensure .env file exists and is properly formatted.')
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentConfigError('AZURE_OPENAI_API_KEY not found in environment variables. Please check your .env file.')
        
        # Initialize generator
        generator = SyntheticDataGenerator(
            config=DEFAULT_CONFIG,
            azure_endpoint="https://iason-gpt-4.openai.azure.com"
        )
        
        # Generate dataset
        logger.info("Generating synthetic dataset...")
        dataset = generator.generate_dataset()
        
        # Export dataset
        generator.export_dataset(dataset)
        
        logger.info("Dataset generation completed!")
        
    except EnvironmentConfigError as e:
        logger.error(f"Environment configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main()