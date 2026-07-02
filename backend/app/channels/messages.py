"""
Shared Kinyarwanda response templates — the channel normalizer referenced
in the architecture doc.

WhatsApp and USSD each translate their own transport format into a call to
app.services.sales_service.record_sale, but before this module existed they
also hand-rolled their own reply copy independently, so the same event
(sale logged, forecast returned) produced different wording depending on
which channel the shopkeeper used. Routing every channel's reply through
these functions keeps that copy in one place.
"""


def sale_logged_message(product_code: str, quantity: float, unit: str) -> str:
    return f"Murakoze! Twanditse: {product_code} {quantity} {unit}."


def sale_not_understood_message() -> str:
    return "Mbabarira, sinabashije gusobanukirwa ubutumwa bwawe. Ongera ugerageze."


def forecast_message(product_code: str, unit: str, predicted_quantity: float | None) -> str:
    if predicted_quantity is None:
        return f"Nta modeli y'{product_code} irahari. Ongera ugerageze nyuma."
    return (
        f"Mu cyumweru gitaha, tubona ko uzagurisha hafi "
        f"{round(predicted_quantity, 1)} {unit} ya {product_code}."
    )
