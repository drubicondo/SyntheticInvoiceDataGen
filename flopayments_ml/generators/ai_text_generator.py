import os
import random
import logging
from typing import Tuple
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime

from ..core.data_models import Fattura, Transazione
from pydantic import BaseModel
from ..core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class AITextGenerator:
    """Handles AI-powered text generation for invoices and transactions"""
    
    transaction_templates = {
        "BONIFICO SEPA": {
            "dettagli": [
                "Bonifico da Voi disposto a favore di: {beneficiary} - Rif. Ord. {order_id}",
                "Bonifico a Vostro favore disposto da: {sender} - Fatt. n. {invoice_num} del {invoice_date}",
                "Bonifico SEPA - Saldo {service_type} Q{quarter}/{year}",
            ],
            "causali": [
                "Pagamento Fattura {invoice_num} - {description}",
                "Acconto Contratto {contract_ref}",
                "Saldo Fatturazione Trimestre Q{quarter}/{year}",
                "Stipendio {month} {year} - Dipendente {employee_id}",
                "Rimborso Spese {travel_description}",
            ]
        },
        "DELEGA F24": {
            "dettagli": [
                "PAGAMENTO F24 - Cod. Tributo {tax_code} - Rif. {ref_month}/{ref_year}",
                "DELEGA F24 - {tax_type} Periodo {month}/{year}",
            ],
            "causali": [
                "Delega F24 - {tax_type} {period}",
                "Versamento {tax_type} {period} - Cod. Tributo {tax_code}",
            ]
        },
        "PAGAMENTO ADUE": {
            "dettagli": [
                "PAGAMENTO ADUE COD. DISP.: {disp_code} NOME: {beneficiary_name}",
                "ADUE - Verso {entity} - Matricola {student_id} - A.A. {academic_year}",
            ],
            "causali": [
                "Pagamento {bill_type} {year} - {installment_info}",
                "Canone {service_type} {year}",
                "Multa Violazione CDS Verbale n. {fine_num}",
            ]
        },
        "SEPA Direct Debit (SDD) BON.UE": {
            "dettagli": [
                "ADDEBITO SDD CORE - Rif. Mandato {mandate_ref} - Fornitore: {supplier}",
                "SDD - Utenza {utility_type} {month} {year}",
            ],
            "causali": [
                "Addebito Utenza Telefonica {month} {year}",
                "Canone Assicurazione Auto Polizza n. {policy_num}",
                "Canone Mensile {service_name}",
            ]
        },
        "ACCR BEU": {
            "dettagli": [
                "ACCR. BEU COD. DISP.: {disp_code} - Bonifico a Vostro favore disposto da: {sender}",
                "ACCREDITO - Rimborso {refund_type} - Periodo {period}",
            ],
            "causali": [
                "Stipendio {month} {year} Dipendente {employee_id}",
                "Rimborso Note Spese Trasferta {destination}",
                "Pagamento Consulenza Progetto {project_id}",
                "Accredito Quota Assicurativa Polizza INAIL - Posizione {position_id}",
            ]
        },
        "COMMISSIONI": {
            "dettagli": [
                "COSTO PER BONIFICO - COMMISSIONI {transaction_code}",
                "COMMISSIONI MENSILI CONTO CORRENTE",
            ],
            "causali": [
                "Commissioni Bancarie",
                "Spese Gestione Conto",
                "Costi Operazione Bancaria",
            ]
        },
        "DOMICILIAZIONI": {
            "dettagli": [
                "PAGAMENTO DOMICILIATO - Utenza {utility_type} {month} {year}",
                "ADDEBITO UTENZA {utility_type} - {billing_period}",
            ],
            "causali": [
                "Pagamento Bolletta {utility_type} {month} {year}",
                "Domiciliazione Rid Utenza Gas",
            ]
        },
        "INCASSO": {
            "dettagli": [
                "VERSAMENTO CONTANTI",
                "INCASSO ASSEGNO CIRCOLARE {check_num}",
            ],
            "causali": [
                "Incasso Versamento Contanti",
                "Accredito Assegno",
            ]
        },
        "ALTRO/GENERICO": {
            "dettagli": [
                "MOVIMENTO VARIO",
                "OPERAZIONE NON CLASSIFICATA",
            ],
            "causali": [
                "Operazione Generica",
                "Movimento Diversi",
            ]
        },
    }

    def __init__(self, azure_endpoint: str, api_version: str = "2024-08-01-preview", 
                 model: str = "gpt-4o", temperature: float = 1.):
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            openai_api_version=api_version,
            deployment_name=model,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=temperature
        )
        self.llm_invoice = self.llm.with_structured_output(Fattura)
        self.llm_trans = self.llm.with_structured_output(Transazione)
    
    def generate_invoice_data(self, company_id: str, data_emissione: str, settore: str, prestatore: str, 
                             importo: float, tipo_servizio: str) -> Tuple[str, str, str]:
        """Generate realistic invoice description and committente"""
        prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Genera una fattura italiana realistica. 
        
        IMPORTANTE per la DESCRIZIONE:
        - Usa un linguaggio tecnico e burocratico tipico delle fatture italiane
        - Sii conciso ma specifico (max 2-3 righe)
        - Includi dettagli tecnici specifici del settore
        - Usa abbreviazioni comuni: "N.", "Rif.", "Cod.", "Art.", "Prot."
        - Includi riferimenti a normative quando appropriato
        - Usa terminologia settoriale specifica
        - Evita descrizioni generiche o marketing
        - Eventuali date menzionate devono essere coerenti con la data di emissione della fattura
        - Il tipo di servizio/prodotto menzionato nella descrizione dev'essere coerente con l'importo della fattura
        
        COMMITTENTE:
        - Genera un nome aziendale italiano realistico
        - Varia tra SRL, SPA, SNCS, SAS, Ditta individuale
        - Il nome dell'azienda dev'essere coerente con il servizio/prodotto menzionato nella descrizione (e.g. una consulenza legale potrebbe essere offerta da uno studio legale)
        """),
        ("human", "Attributi fattura:\n{attributi_fattura}")
        ])
        
        attributi_fattura = f"""
        - Data emissione: {data_emissione}
        - Settore: {settore}
        - Prestatore: {prestatore}
        - Importo: €{importo:.2f}
        - Tipo servizio: {tipo_servizio}
        """
        
        chain = prompt_template | self.llm_invoice
        try:
            response: AIInvoiceOutput = chain.invoke({"attributi_fattura": attributi_fattura})
            return response.descrizione, response.committente, response.numero_fattura
        except Exception as e:
            logger.error(f"Error generating invoice data: {e}")
            return self._get_fallback_invoice_data(tipo_servizio, data_emissione)
    
    def generate_transaction_data(self, fattura: Fattura, importo: float,
                                invoice_number_probability: float = 0.1) -> Tuple[str, str, str, bool]:
        """Generate realistic transaction dettaglio, causale and controparte"""
        include_invoice_number = random.random() < invoice_number_probability
    
        # Determine the prompt content based on whether to include the invoice number
        invoice_number_instruction = ""
        if include_invoice_number:
            invoice_number_instruction = "- Includi il numero fattura nel dettaglio e/o causale"
        else:
            invoice_number_instruction = "- NON includere il numero fattura - usa solo descrizioni generiche del servizio"
    
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""Genera una transazione bancaria italiana realistica.
    
            IMPORTANTE:
            - Il dettaglio deve essere tipico dei bonifici italiani: qui deve comparire anche la tipologia di pagamento che dev'essere del tipo bonifico SEPA, Deleghe F24, pagamento ADUE, Sepa Direct Debit (SDD) BON.UE, ACCR BEU, etc. coerentemente con la tipologia di fattura
            - La causale deve essere concisa e professionale e può dipendere dalla tipologia di pagamento. Di seguito alcuni esempio:
                Bonifici SEPA:
                    Bonifico SEPA – Fattura 2025/117 – Fornitura software SaaS – Cliente: Rossi S.r.l. – ID Ord. A456
                    Acconto contratto fornitura servizi IT
                    Saldo fatturazione trimestre Q1/2024
                Deleghe F24:
                    Delega F24 – Cod. tributo 4034 – Rata acconto IVA 2025 – Agenzia Entrate Roma 3 – Mese rif. 07/2025
                    Versamento IRPEF periodo gennaio 2024
                    IVA mensile marzo 2024 - cod. tributo 6001
                    Acconto IRES 2024 - prima rata
                    Pagamento contributi INPS dipendenti febbraio 2024
                pagamento ADUE:
                    ADUE – Verso MEF – Diritti di segreteria universitaria – Matricola T005678 – A.A. 2024/25 – Scad. 31/07/2025
                    Pagamento TARI 2024 - prima rata
                    Canone occupazione suolo pubblico 2024
                    Multa violazione CDS verbale n. 123456
                SEPA Direct Debit (SDD) BON.UE:
                    Addebito utenza telefonica marzo 2024
                    Canone assicurazione auto polizza n. 987654
                    SDD CORE – Rif. Mandato IT32XYZ000123 – Canone mensile Cloud Hosting – Addebito BON.UE – Fornitore: NetService S.p.A.
                ACCR BEU (Accredito Beneficiario Europeo):
                    Stipendio marzo 2024 dipendente IT001
                    Rimborso note spese trasferta Germania
                    Pagamento consulenza progetto EU2024
                    ACCR BEU – Rimborso quota assicurativa polizza INAIL – Periodo 07/2025 – Posizione 0123456/7 – Addebito su c/c
            - La controparte può essere uguale o leggermente diversa dal beneficiario
            - Usa terminologia bancaria italiana standard
            {invoice_number_instruction}
            """),
            ("human", "Attributi transazione:\n{attributi_transazione}")
        ])
    
        # Prepare transaction attributes based on whether to include invoice number
        if include_invoice_number:
            attributi_transazione = f"""
            BENEFICIARIO: {fattura.prestatore}
            IMPORTO: €{importo:.2f}
            NUMERO_FATTURA: {fattura.numero_fattura}
            DESCRIZIONE_FATTURA: {fattura.descrizione[:100]}...
            """
        else:
            attributi_transazione = f"""
            BENEFICIARIO: {fattura.prestatore}
            IMPORTO: €{importo:.2f}
            DESCRIZIONE_FATTURA: {fattura.descrizione[:100]}...
            TIPO_SERVIZIO: {getattr(fattura, 'tipo_servizio', 'Servizio professionale')}
            """
    
        chain = prompt_template | self.llm_trans
    
        try:
            response: Transazione = chain.invoke({"attributi_transazione": attributi_transazione})
            return response.dettaglio, response.causale, response.controparte, include_invoice_number
        except Exception as e:
            logger.error(f"Error generating transaction data: {e}")
            return self._get_fallback_transaction_data(fattura, include_invoice_number)
    
    def _get_fallback_invoice_data(self, tipo_servizio: str, data_emissione) -> Tuple[str, str, str]:
        """Generate fallback invoice data when AI generation fails"""
        fallback_descriptions = {
            "trasporto": "Servizi trasporto merci c/terzi - Rif. DDT N. 125/2024",
            "consulting": "Attività consulenza specialistica - Prot. N. 456/2024",
            "formazione": "Corso formazione sicurezza D.Lgs 81/08 - N. 12 ore",
            "manutenzione": "Intervento manutenzione ordinaria impianti",
            "pulizia": "Servizi pulizia locali - Periodo 01/07-31/07/2024"
        }
        fallback_desc = fallback_descriptions.get(
            tipo_servizio.lower(), 
            f"Prestazione {tipo_servizio} - Rif. contratto"
        )
        # Ensure data_emissione is a datetime object
        if isinstance(data_emissione, str):
            try:
                data_emissione_dt = datetime.strptime(data_emissione, "%Y-%m-%d")
            except Exception:
                data_emissione_dt = datetime.now()
        else:
            data_emissione_dt = data_emissione
        fallback_numero = f"FT{data_emissione_dt.year}/{random.randint(1000, 9999)}"
        return fallback_desc, "BETA SOLUTIONS SRL", fallback_numero
    
    def _get_fallback_transaction_data(self, fattura: Fattura, include_invoice_number: bool) -> Tuple[str, str, str, bool]:
        """Generate fallback transaction data when AI generation fails"""
        if include_invoice_number:
            fallback_dettaglio = f"BONIFICO SEPA - Pagamento fattura n. {fattura.numero_fattura}"
            fallback_causale = f"Pagamento fattura {fattura.numero_fattura}"
        else:
            service_type = getattr(fattura, 'tipo_servizio', 'servizi')
            fallback_dettaglio = f"BONIFICO SEPA - Pagamento {service_type}"
            fallback_causale = f"Pagamento {service_type}"

        fallback_controparte = fattura.prestatore
        return fallback_dettaglio, fallback_causale, fallback_controparte, include_invoice_number

    def generate_invoice_data_batch(self, invoices: list[dict]) -> list[Tuple[str, str, str]]:
        """Generate invoice texts for a batch of invoices."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Genera una fattura italiana realistica.

            IMPORTANTE per la DESCRIZIONE:
            - Usa un linguaggio tecnico e burocratico tipico delle fatture italiane
            - Sii conciso ma specifico (max 2-3 righe)
            - Includi dettagli tecnici specifici del settore
            - Usa abbreviazioni comuni: \"N.\", \"Rif.\", \"Cod.\", \"Art.\", \"Prot.\"
            - Includi riferimenti a normative quando appropriato
            - Usa terminologia settoriale specifica
            - Evita descrizioni generiche o marketing
            - Eventuali date menzionate devono essere coerenti con la data di emissione della fattura
            - Il tipo di servizio/prodotto menzionato nella descrizione dev'essere coerente con l'importo della fattura

            COMMITTENTE:
            - Genera un nome aziendale italiano realistico
            - Varia tra SRL, SPA, SNCS, SAS, Ditta individuale
            - Il nome dell'azienda dev'essere coerente con il servizio/prodotto menzionato nella descrizione
            """),
            ("human", "Attributi fattura:\n{attributi_fattura}")
        ])

        chain = prompt_template | self.llm_invoice
        inputs = []
        for inv in invoices:
            attributi = f"""
            - Data emissione: {inv['data_emissione']}
            - Settore: {inv['settore']}
            - Prestatore: {inv['prestatore']}
            - Importo: €{inv['importo']:.2f}
            - Tipo servizio: {inv['tipo_servizio']}
            """
            inputs.append({"attributi_fattura": attributi})

        try:
            responses = chain.batch(inputs)
            return [(r.descrizione, r.committente, r.numero_fattura) for r in responses]
        except Exception as e:
            logger.error(f"Error generating invoice batch: {e}")
            return [self._get_fallback_invoice_data(i['tipo_servizio'], i['data_emissione']) for i in invoices]

    def generate_transaction_data_batch(self, transactions: list[dict]) -> list[Tuple[str, str, str, bool]]:
        """Generate transaction texts for a batch of payments."""
        prompt_with = ChatPromptTemplate.from_messages([
            ("system", """Genera una transazione bancaria italiana realistica.

            IMPORTANTE:
            - Il dettaglio deve essere tipico dei bonifici italiani
            - La causale deve essere concisa e professionale
            - La controparte può essere uguale o leggermente diversa dal beneficiario
            - Usa terminologia bancaria italiana standard
            - Include il numero fattura nel dettaglio e/o causale
            """),
            ("human", "Attributi transazione:\n{attributi_transazione}")
        ])

        prompt_without = ChatPromptTemplate.from_messages([
            ("system", """Genera una transazione bancaria italiana realistica.

            IMPORTANTE:
            - Il dettaglio deve essere tipico dei bonifici italiani
            - La causale deve essere concisa e professionale
            - La controparte può essere uguale o leggermente diversa dal beneficiario
            - Usa terminologia bancaria italiana standard
            - NON includere il numero fattura - usa solo descrizioni generiche del servizio
            """),
            ("human", "Attributi transazione:\n{attributi_transazione}")
        ])

        chain_with = prompt_with | self.llm_trans
        chain_without = prompt_without | self.llm_trans

        inputs_with, inputs_without = [], []
        idx_with, idx_without = [], []
        for idx, t in enumerate(transactions):
            fattura = t['fattura']
            attrib = f"""
            BENEFICIARIO: {fattura.prestatore}
            IMPORTO: €{t['importo']:.2f}
            {f'NUMERO_FATTURA: {fattura.numero_fattura}' if t['include_invoice_number'] else ''}
            DESCRIZIONE_FATTURA: {fattura.descrizione[:100]}...
            TIPO_SERVIZIO: {getattr(fattura, 'tipo_servizio', 'Servizio professionale')}
            """
            if t['include_invoice_number']:
                inputs_with.append({"attributi_transazione": attrib})
                idx_with.append(idx)
            else:
                inputs_without.append({"attributi_transazione": attrib})
                idx_without.append(idx)

        results = [None] * len(transactions)
        try:
            if inputs_with:
                res_with = chain_with.batch(inputs_with)
                for i, r in zip(idx_with, res_with):
                    results[i] = (r.dettaglio, r.causale, r.controparte, True)
            if inputs_without:
                res_without = chain_without.batch(inputs_without)
                for i, r in zip(idx_without, res_without):
                    results[i] = (r.dettaglio, r.causale, r.controparte, False)
        except Exception as e:
            logger.error(f"Error generating transaction batch: {e}")
            for i, t in enumerate(transactions):
                results[i] = self._get_fallback_transaction_data(t['fattura'], t['include_invoice_number'])

        return results
