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


def sale_logged_message(product_code: str, quantity: float, unit: str, locale: str = "kinyarwanda") -> str:
    if locale == "english":
        return f"Thank you! Recorded: {product_code} {quantity} {unit}."
    return f"Murakoze! Twanditse: {product_code} {quantity} {unit}."


def sale_not_understood_message(locale: str = "kinyarwanda") -> str:
    if locale == "english":
        return "Sorry, we couldn't understand your message. Please try again."
    return "Mbabarira, sinabashije gusobanukirwa ubutumwa bwawe. Ongera ugerageze."


def forecast_message(
    product_code: str, unit: str, predicted_quantity: float | None, locale: str = "kinyarwanda"
) -> str:
    if predicted_quantity is None:
        if locale == "english":
            return f"No model for {product_code} available yet. Try again later."
        return f"Nta modeli y'{product_code} irahari. Ongera ugerageze nyuma."
    if locale == "english":
        return (
            f"Next week, we predict you'll sell about "
            f"{round(predicted_quantity, 1)} {unit} of {product_code}."
        )
    return (
        f"Mu cyumweru gitaha, tubona ko uzagurisha hafi "
        f"{round(predicted_quantity, 1)} {unit} ya {product_code}."
    )


def recent_sales_message(sales: list, locale: str = "kinyarwanda") -> str:
    """`sales` is a list of SalesLog rows, most recent first (see
    app.services.sales_service.get_recent_sales) -- kept as plain attribute
    access rather than importing the ORM model here, so this stays a pure
    text-formatting function like the rest of the module."""
    if not sales:
        if locale == "english":
            return "You haven't logged any sales yet."
        return "Ntabwo waragurisha kintu na kimwe."
    header = "Your recent sales:" if locale == "english" else "Amagurisha yanyu aheruka:"
    lines = [f"{s.logged_at.strftime('%m/%d')} {s.product_code} {s.quantity:g}{s.unit}" for s in sales]
    return header + "\n" + "\n".join(lines)
