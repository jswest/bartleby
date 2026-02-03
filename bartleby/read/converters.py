"""HTML to PDF conversion using Playwright."""

from pathlib import Path
from playwright.sync_api import sync_playwright
from loguru import logger


def convert_html_to_pdf(html_path: Path, output_dir: Path) -> Path:
    """
    Convert an HTML file to PDF using Playwright.

    Args:
        html_path: Path to the HTML file to convert
        output_dir: Directory where the PDF should be saved

    Returns:
        Path to the generated PDF file

    Raises:
        FileNotFoundError: If the HTML file doesn't exist
        RuntimeError: If PDF conversion fails
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    if not html_path.is_file():
        raise ValueError(f"Path is not a file: {html_path}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output PDF path (same name as HTML but with .pdf extension)
    pdf_path = output_dir / f"{html_path.stem}.pdf"

    try:
        logger.debug(f"Converting HTML to PDF: {html_path} -> {pdf_path}")

        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch()
            page = browser.new_page()

            # Load HTML file
            html_url = f"file://{html_path.absolute()}"
            page.goto(html_url, wait_until="networkidle")

            # Convert to PDF with Letter size (8.5 x 11 inches)
            page.pdf(
                path=str(pdf_path),
                format="Letter",  # 8.5 x 11 inches
                print_background=True,  # Include background colors/images
                margin={
                    "top": "0.5in",
                    "right": "0.5in",
                    "bottom": "0.5in",
                    "left": "0.5in",
                }
            )

            browser.close()

        logger.debug(f"Successfully converted {html_path.name} to PDF")
        return pdf_path

    except Exception as e:
        logger.error(f"Failed to convert HTML to PDF: {e}")
        raise RuntimeError(f"PDF conversion failed for {html_path}: {e}") from e
