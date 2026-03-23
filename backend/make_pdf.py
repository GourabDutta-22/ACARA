from reportlab.pdfgen import canvas
c = canvas.Canvas("test.pdf")
c.drawString(100, 750, "Hello World from PDF testing.")
c.save()
