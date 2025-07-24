from datetime import datetime, timedelta
from uuid import UUID, uuid4
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    NonNegativeFloat,
    ValidationError,
    FieldValidationInfo,
    field_validator,
)
from typing import Literal, Optional


class Fattura(BaseModel):
    """Modello dati per una fattura"""
    id: Optional[UUID] = Field(default_factory=uuid4, description="Unique identifier")
    data_emissione: datetime = Field(
        ...,
        examples = ['01-07-2025', '01-05-2024', '10/12/2024', '09/01/2024'],
        description = "data di emissione fattura"
    )
    data_scadenza: datetime = Field(
        ...,
        examples = ['10-08-2025', '09-07-2024', '31/12/2024', '12/03/2024'],
        description = "data di emissione fattura"
    )
    numero_fattura: str = Field(
        ...,
        examples = ['FT/2024/001', '2025/123', 'FV-00045', 'FATT2024-0001', '001/2024'],
        description = "Numero identificativo univoco della fattura"
    )
    descrizione: str = Field(
        ...,
        examples = [
        "N. 24 MODULI DI CORSI DI FORMAZIONE GENERALE E SPECIFICA RISCHIO BASSO  SECONDO L'ACCORDO STATO-REGIONI DEL 21/12/2011 IN MOD. E-LEARNING  NOSTRE COORDINATE BANCARIE: BPER BANCA - AGENZIA DI VEDANO AL LAMBRO (MB) C/C INTESTATO A SERVIZI AZIENDALI INTEGRATI S.A.I. SRL IBAN  IT09C0538734000000042628500", 
        "Canone Locazione Finanziaria - Scadenza del 28/11/2024. rata nr.  35",
        "Avviso di parcella n. 353 del 30/09/2024",
        "restauro oggetti di pelletteria da ufficio",
        "FORNITURA ENERGIA ELETTRICA"
        ],
        description = "descrizione dei prodotti/servizi"
    )
    importo: PositiveFloat = Field(
        ...,
        examples = [12338.90, 890.60, 1726.69, 450.00],
        description = "importo della fattura"
    )
    prestatore: str = Field(
        ...,
        examples = ['DigitEd S.p.A. a Socio unico', 'Jakala S.p.A. S.B.', 'Intesa Sanpaolo S.p.a', 'ANGELO MARIO ROVERSI', 'EDENRED ITALIA Srl'],
        description = "soggetto che esegue la prestazione di servizio"
    )
    committente: str = Field(
        ...,
        examples = ['AESON SRL'],
        description = "Soggetto che richiede la prestazione di servizio"
    )

    @field_validator('data_scadenza')
    @classmethod
    def validate_scadenza_date(
        cls, v: datetime, info: FieldValidationInfo
    ) -> datetime:
        # Access the entire model's values to get data_emissione
        if info.data and 'data_emissione' in info.data:
            data_emissione = info.data['data_emissione']
            if not isinstance(data_emissione, datetime):
                raise ValueError("data_emissione must be a datetime object")
        
            # Calculate the minimum allowed data_scadenza (same day as data_emissione)
            min_data_scadenza = data_emissione.replace(hour=0, minute=0, second=0, microsecond=0)
        
            # Calculate the maximum allowed data_scadenza (90 days after data_emissione)
            max_data_scadenza = data_emissione + timedelta(days=90)
        
            # Check if data_scadenza is within the allowed range
            if not (min_data_scadenza <= v <= max_data_scadenza):
                raise ValueError(
                    f"La data di scadenza ({v.strftime('%Y-%m-%d')}) deve essere compresa tra "
                    f"la data di emissione ({data_emissione.strftime('%Y-%m-%d')}) "
                    f"e al massimo 90 giorni dopo ({max_data_scadenza.strftime('%Y-%m-%d')})."
                )
        return v


class Transazione(BaseModel):
    """Modello dati per una transazione bancaria"""
    id: Optional[UUID] = Field(default_factory=uuid4, description="Unique identifier")
    data: datetime = Field(
        ...,
        examples = [datetime(2025, 7, 1), datetime(2025, 6, 15)],
        description = "data del pagamento"
    )
    dettaglio: str = Field(

        ...,
        examples = [
            "COSTO PER BONIFICO Bonifico da Voi disposto a favore di: BENEFICIARI DIVERSI - COMMISSIONI W0240257615359022500000026 0125021220300042",
            "PAGAMENTO ADUE COD. DISP.: 0125022556241450 NOME: A2A S P A MANDATO: S9335009557",
            "ACCR. BEU COD. DISP.: 0125030736616998 CASH Q0G8KCGGBJ5IA17413395373340.6704330 Fatt. n. 7 del 07/03/2025 Bonifico a Vostro favore disposto da: MITT.: IASON ITALIA BENEF.: AESON SRL BIC. ORD.: BCITITMM",
            "DISP.BEU STIP. TOTALE NUMERO BONIFICI: 4 TOTALE IMPORTO BONIFICI: 6.235,00 0125030737683077 FEBBRAIO 2025 Bonifico da Voi disposto a favore di: BENEFICIARI DIVERSI -",
            "DISP.BEU STIP. TOTALE NUMERO BONIFICI: 5 TOTALE IMPORTO BONIFICI: 5.622,55 0125020603910797 GENNAIO 2025 Bonifico da Voi disposto a favore di: BENEFICIARI DIVERSI -",
            "CANONE MENSILE CANONE MENSILE MESE DI FEBBRAIO",
            "BON.UE CAN.TELEM. 0125020559399115 W0240257615359022500000017 Bonifico da Voi disposto a favore di: Edenred Italia S.r.l. CODICE CLIENTE 899462 e P.IVA 10550680960 - ordine n. 3 del 05/02/2025"
        ],
        description = "dettagli di riferimento relativi a un pagamento o a una transazione bancaria"
    )
    importo: float = Field(
        ...,
        examples = [-12338, 12132.94, 22737, -123.00, 2132.23],
        description = "importo della transazione"
    )
    tipologia_movimento: Literal[
        'commissioni',
        'domiciliazioni',
        'fisco',
        'incasso',
        'pagamento'
    ] = Field(
        ...,
        description = "tipologia della transazione"
    )
    controparte: str = Field(
        ...,
        examples = [
            "IASON ITALIA",
            "NOT_SPECIFIED",
            "Dipendenti",
            "RAG. AGOSTINI ROBERTO",
            "Olivo Luca",
            "Studio Roversi",
            "Amundi SGR S.p.A",
            "SARTORIA ESTHES SRL",
            "SERVIZI AZIENDALI INTEGRATI - S.A.I"
        ],
        description = "controparte della transazione" 
    )
    causale: str = Field(
        ...,
        examples = [
            "stripendi",
            "Fatt. n. 2 del 28/02/2025",
            "CODICE CLIENTE 899462 e P.IVA 10550680960 - ordine n. 3 del 05/02/2025",
            "Codice Azienda 0000066831 - AESON SRL 10550680960 - Anno 2025",
            "Finanziamento socio",
            "NOT_SPECIFIED"
        ],
        description = "causale della transazione"
    )
    invoice_number: bool #whether or not the descrizione/causal contains invoice number
    