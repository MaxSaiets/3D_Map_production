import type { LegalSet } from "./content";

export const en: LegalSet = {
  offer: {
    title: "Public Offer Agreement",
    intro: [
      "This document is an official public proposal (offer) by {ownerFull} (hereinafter the \"Seller\") to conclude a distance contract for the sale of goods and provision of services on the terms set out below, in accordance with Articles 633, 641 and 642 of the Civil Code of Ukraine and the Laws of Ukraine \"On Electronic Commerce\" and \"On Protection of Consumer Rights\".",
      "By placing and/or paying for an order on the {domain} website (hereinafter the \"Website\"), the Buyer confirms that they have fully read, understood and unconditionally accepted the terms of this Agreement (acceptance of the offer). Acceptance of this offer is equivalent to concluding a contract in written form.",
    ],
    sections: [
      { h: "1. Definitions", blocks: [
        { kv: [
          { k: "Seller", v: "{ownerFull}, Tax ID (RNOKPP) {taxId}, single tax payer." },
          { k: "Buyer", v: "any legally capable natural person or legal entity who has placed an order on the Website and accepted this offer." },
          { k: "Website", v: "the {storeName} online store at {domain}, including the online 3D model builder." },
          { k: "Builder", v: "the Website's software service in which the Buyer independently selects the parameters of the future product: map area, shape, size, format, engraving text, etc." },
          { k: "Goods", v: "a physical product (a 3D-printed city map, wall panel, map keychain, magnet, etc.) made of Eco PLA bioplastic, manufactured to the Buyer's individual order." },
          { k: "Digital content", v: "a generated digital 3D model (a 3MF/STL file) suitable for self-service 3D printing." },
          { k: "Order", v: "the Buyer's request, placed through the Website, to purchase Goods and/or Digital content." },
        ] },
      ] },
      { h: "2. Subject of the Agreement", blocks: [
        { p: "The Seller undertakes, at the Buyer's request, to generate a digital 3D model and/or manufacture a physical product according to the Buyer's individual parameters and transfer it to the Buyer, and the Buyer undertakes to accept and pay for the Goods/Digital content on the terms of this Agreement." },
        { p: "Each product is manufactured individually according to the parameters (location on the map, shape, size, text, colors) that the Buyer selects independently in the Builder. Accordingly, the Goods are products made to the Buyer's individual order within the meaning of the Law of Ukraine \"On Protection of Consumer Rights\"." },
        { p: "The range, characteristics and indicative prices of the Goods are listed on the Website. Photographs and 3D previews are for information purposes: the finished product may have minor differences in material shade and layer texture, which is a natural feature of 3D printing technology and does not constitute a defect of the Goods." },
      ] },
      { h: "3. Placing an Order", blocks: [
        { ul: [
          "The Buyer independently creates the model in the Builder: selects the city/map area, shape, size, format (relief/flat, magnet) and, optionally, engraving text or a GPX route.",
          "Before checkout, the Buyer sees a 3D preview of the model and the final cost of the order.",
          "To place the order, the Buyer provides contact details: name, phone, email if needed, delivery method and the carrier's address/branch.",
          "The order is deemed accepted for fulfillment after it has been paid online or confirmed by an operator (by arrangement).",
        ] },
        { p: "The Buyer is solely responsible for the correctness of the selected model parameters (map area, text, size) and the accuracy of the contact details provided. The Seller bears no responsibility for the consequences of errors in the data supplied by the Buyer, including the manufacture of a product with erroneous text or map area that the Buyer themselves approved in the Builder." },
      ] },
      { h: "4. Price and Payment", blocks: [
        { p: "Prices on the Website are shown in hryvnia (for orders within Ukraine) and in euros (indicative). The Buyer sees the final cost of the order at the checkout step, before payment. Delivery costs are not included in the price of the Goods and are paid separately at the carrier's tariffs." },
        { p: "Payment is made online via the LiqPay payment service (JSC CB \"PrivatBank\"): by Visa/Mastercard bank card and other methods available in LiqPay. Payment card data is processed on the payment system's side; the Seller does not receive or store it." },
        { p: "By arrangement with an operator, payment by another agreed method is possible. Downloading finished Digital content within the account's free limit is free of charge." },
        { p: "The Goods are paid for in full before being sent to production, unless otherwise agreed by the parties. Details are on the [delivery:\"Payment and Delivery\"] page." },
      ] },
      { h: "5. Production Time", blocks: [
        { p: "Digital content is generated automatically and provided in the Buyer's account / by email immediately after generation or after the order is confirmed." },
        { p: "The estimated production time for a physical product is 1–3 business days from payment/confirmation of the order. In the event of high workload or technical complexity of the product, this time may be extended, of which the Seller notifies the Buyer." },
      ] },
      { h: "6. Delivery", blocks: [
        { p: "Delivery within Ukraine is carried out by Nova Poshta (branch, parcel locker) or Ukrposhta. The estimated delivery time within Ukraine is 2–4 business days after dispatch." },
        { p: "Delivery costs are calculated at the carrier's tariffs and paid by the Buyer separately (usually on receipt). Title to the Goods and the risk of accidental damage pass to the Buyer upon receipt of the Goods from the carrier." },
        { p: "Upon receipt, the Buyer must inspect the shipment for the integrity of the packaging and the product. In the event of damage in transit, it should be documented in the carrier's damage report and the Seller notified — such cases are resolved by a free reprint or a refund (see Section 8)." },
      ] },
      { h: "7. Quality and Warranty", blocks: [
        { p: "The Seller guarantees that the product conforms to the parameters approved by the Buyer in the Builder and that the print quality is proper. Physical products carry a 60-day warranty from the moment of receipt covering printing defects and delamination not caused by the Buyer." },
        { p: "Natural features of FDM printing technology (visible print layers, slight differences in plastic shade between batches, process marks on the bottom surface) are not defects of the Goods." },
        { p: "The Seller is not responsible for the print result on the Buyer's own equipment when the Buyer prints the downloaded Digital content themselves (the quality of such printing depends on the Buyer's printer, material and settings)." },
      ] },
      { h: "8. Refunds and Exchanges", blocks: [
        { p: "Since the Goods are manufactured to individual order according to the Buyer's unique parameters, Goods of proper quality are not subject to return or exchange (Law of Ukraine \"On Protection of Consumer Rights\"; the list of goods approved by Resolution of the Cabinet of Ministers of Ukraine No. 172 of 19.03.1994). Once access to the download has been provided, Digital content is non-refundable as electronic content the service for which has been consumed." },
        { ul: [
          "Before the order is sent to production, the Buyer may cancel it and receive a full refund.",
          "In the event of a defect, damage in transit or non-conformity of the product with the approved parameters, the Seller, at the Buyer's choice, manufactures and sends a new product free of charge or refunds the full price.",
          "If the Digital content is technically corrupted or fails to download through the Seller's fault, the file is regenerated free of charge or the money is refunded.",
        ] },
        { p: "The procedure and timeframes are described in the [refund:\"Refunds and Exchanges\"] document, which forms an integral part of this Agreement. Refunds are made using the same method by which payment was made (to the card via LiqPay), within the timeframes set by the rules of the payment system and the bank." },
      ] },
      { h: "9. Rights and Obligations of the Parties", blocks: [
        { p: "The Seller shall: manufacture the Goods in accordance with the parameters approved by the Buyer; meet the stated deadlines; keep the Buyer informed of the order status; ensure the confidentiality of the Buyer's personal data." },
        { p: "The Seller may: engage third parties to perform its obligations (carriers, payment services); suspend fulfillment of an order if it has not been paid for; refuse to manufacture a product whose content violates the laws of Ukraine (in particular, contains prohibited symbols or hate speech), with a full refund." },
        { p: "The Buyer shall: provide accurate data for fulfilling the order; pay for the order; collect the Goods from the carrier within the shipment storage period." },
        { p: "The Buyer may: receive Goods of proper quality within the stated time; receive information about the status of their order; file a complaint in the manner provided for by this Agreement." },
      ] },
      { h: "10. Intellectual Property Rights", blocks: [
        { p: "Cartographic data is provided by OpenStreetMap (© OpenStreetMap contributors, ODbL license); elevation data comes from open sources. The generated 3D model is provided to the Buyer for personal, non-commercial use and printing. Large-scale commercial use or resale of the models requires a separate written arrangement with the Seller." },
        { p: "A GPX route uploaded by the Buyer is the Buyer's own data and is processed solely to build their model (see the [privacy:Privacy Policy]). The Buyer warrants that the engraving text they order does not infringe the rights of third parties." },
      ] },
      { h: "11. Personal Data", blocks: [
        { p: "By placing an order, the Buyer consents to the processing of their personal data (name, contact details, delivery address) solely for the purpose of performing this Agreement, in accordance with the Law of Ukraine \"On Personal Data Protection\" and the [privacy:Privacy Policy]. The data is not shared with third parties except where necessary to fulfill the order (carrier, payment service)." },
      ] },
      { h: "12. Liability and Force Majeure", blocks: [
        { p: "For non-performance or improper performance of obligations, the parties bear liability in accordance with the applicable laws of Ukraine. The Seller's aggregate liability under any claims is limited to the amount actually paid by the Buyer for the order." },
        { p: "The parties are released from liability for full or partial non-performance of obligations if it resulted from force majeure circumstances: military action, shelling, power outages, natural disasters, decisions of public authorities, carrier disruptions, etc. Deadlines for performance are extended for the duration of such circumstances." },
      ] },
      { h: "13. Complaints and Dispute Resolution", blocks: [
        { p: "Complaints regarding an order are accepted at {email} or by phone {phone}, stating the order number. The Seller reviews requests within 1–3 business days. Disputes are resolved through negotiation and, failing agreement, in the manner established by the applicable laws of Ukraine." },
      ] },
      { h: "14. Term and Amendment of the Terms", blocks: [
        { p: "The Agreement enters into force upon the Buyer's acceptance of the offer and remains in effect until the parties have fully performed their obligations. The Seller may amend the terms of this offer by publishing a new version on this page; the new version applies to orders placed after its publication. The current version is permanently available at {domain}/offer." },
        { p: "The following documents form integral parts of this Agreement: [refund:\"Refunds and Exchanges\"], [delivery:\"Payment and Delivery\"], [privacy:\"Privacy Policy\"] and [terms:\"Terms of Use\"]." },
      ] },
      { h: "15. Seller's Details", blocks: [
        { kv: [
          { k: "Seller", v: "{ownerFull}" },
          { k: "Tax ID (RNOKPP)", v: "{taxId}" },
          { k: "Type of activity (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Store", v: "{storeName}, {storeAddress}" },
          { k: "Email", v: "{email}" },
          { k: "Phone", v: "{phone}" },
        ] },
      ] },
    ],
  },

  refund: {
    title: "Refunds and Exchanges",
    sections: [
      { h: "Nature of the Goods", blocks: [
        { p: "All {storeName} products are manufactured individually according to parameters that the Buyer selects themselves (location on the map, shape, size, text), and the digital 3D models are electronic content. This affects the refund terms under Ukraine's Law on Consumer Rights Protection (goods of proper quality made to individual order are not subject to return or exchange)." },
      ] },
      { h: "Digital Files (3MF/STL Downloads)", blocks: [
        { p: "Funds for a digital file are non-refundable once the file has been generated and made available for download, as the service has already been provided in full. If the file fails to download for technical reasons or is corrupted on our side, we will regenerate it free of charge or refund your money." },
      ] },
      { h: "Print to Order (Physical Product)", blocks: [
        { ul: [
          "Before production begins — you may cancel the order and receive a full refund, provided we have not yet started printing.",
          "After production has begun — funds are non-refundable, as the product is manufactured personally for your order.",
          "Defects, damage in transit or non-conformity with the order — we will reprint and resend the product free of charge or refund the full amount (at your choice).",
        ] },
      ] },
      { h: "How to Request a Refund", blocks: [
        { p: "Write to {email} or call {phone}, providing the order number, the reason and (for defects) a photo of the product. We will review your request within 1–3 business days." },
        { p: "Refunds are made using the same method by which payment was made (a refund to the card via LiqPay), within the timeframes set by the rules of the payment system and the bank." },
      ] },
      { h: "Contact", blocks: [
        { p: "This document forms an integral part of the [offer:Public Offer Agreement]. Questions: {email}." },
      ] },
    ],
  },

  delivery: {
    title: "Payment and Delivery",
    sections: [
      { h: "Products and Prices", blocks: [
        { p: "Product prices (delivery is charged separately by the carrier):" },
        { ul: [
          "Map keychain — from 120 ₴ (≈ 3 €).",
          "3D district map: S 5.5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴ (maps — from 6 €).",
          "Fridge magnet (map) — 150 ₴.",
          "Downloading the finished 3MF/STL file for self-printing — free within your account limit.",
        ] },
      ] },
      { h: "Payment", blocks: [
        { p: "Online payment by Visa / Mastercard bank card through the secure LiqPay service. Card data is processed on the payment system's side — we do not store it. Payment by arrangement with an operator is also possible. The digital file within the free limit is provided at no charge." },
      ] },
      { h: "Manufacture", blocks: [
        { p: "Products are printed to order from Eco PLA bioplastic. The estimated production time is 1–3 business days after the order is confirmed (depending on workload and complexity)." },
      ] },
      { h: "Delivery", blocks: [
        { p: "Ukraine:" },
        { ul: ["Nova Poshta — branch or parcel locker.", "Ukrposhta — branch."] },
        { p: "Delivery cost is calculated according to the carrier's tariffs and paid separately (usually on receipt). The estimated delivery time within Ukraine is 2–4 business days after dispatch." },
      ] },
      { h: "Refunds", blocks: [
        { p: "Refund terms are described on the [refund:\"Refunds and Exchanges\"] page. General terms are in the [offer:Public Offer Agreement]." },
      ] },
      { h: "Contact", blocks: [
        { p: "Questions about payment or delivery: {email}, {phone}." },
      ] },
    ],
  },

  contacts: {
    title: "Contacts and Details",
    sections: [
      { h: "Contact Us", blocks: [
        { kv: [
          { k: "Email", v: "{email}" },
          { k: "Phone", v: "{phone}" },
          { k: "Website", v: "{domain}" },
        ] },
        { p: "Order processing hours: daily, we respond within 24 hours." },
      ] },
      { h: "Store", blocks: [
        { kv: [
          { k: "Name", v: "{storeName}" },
          { k: "Store address", v: "{storeAddress}" },
        ] },
      ] },
      { h: "Seller (Sole Proprietor)", blocks: [
        { kv: [
          { k: "Name", v: "{ownerFull}" },
          { k: "Tax ID", v: "{taxId}" },
          { k: "Type of activity (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
        ] },
      ] },
      { h: "What We Sell", blocks: [
        { p: "{storeName} offers 3D models of city maps and map keychains made to order. You can download a ready-to-print file (3MF/STL) within the free limit, or order a product printed from Eco PLA bioplastic with delivery. Indicative prices are shown in the designer and on the [delivery:\"Payment and Delivery\"] page." },
      ] },
      { h: "Documents", blocks: [
        { ul: [
          "[offer:Public Offer Agreement]",
          "[refund:Refunds and Exchanges]",
          "[delivery:Payment and Delivery]",
          "[privacy:Privacy Policy]",
          "[terms:Terms of Use]",
        ] },
      ] },
    ],
  },

  privacy: {
    title: "Privacy Policy",
    sections: [
      { h: "What Data We Collect", blocks: [
        { p: "Your name, email or phone number (when you sign in / register), as well as the contact details and delivery address you provide when placing an order. Technical data (the history of generated models) is stored in your account." },
      ] },
      { h: "How We Use the Data", blocks: [
        { p: "Solely to provide the service: signing in to your account, saving your model history, processing and delivering orders, and contacting you about your order. We do not sell or share your data with third parties for advertising." },
      ] },
      { h: "Uploaded Routes (GPX) and Geodata", blocks: [
        { p: "If you upload a GPX file (for example, an export of your own activity from Strava or another app), we process the route coordinates solely to build your 3D model. This is your own data — we do not publish it, share it with third parties or use it for advertising. The route points are simplified and stored only for as long as needed to generate the model and (if you are logged in) to maintain your model history in your account; you can ask us to delete them at any time." },
        { p: "Searching for a place on the map sends your query to the OpenStreetMap geocoding service (Nominatim), and the maps themselves are loaded from OpenStreetMap tiles — in accordance with their terms of use. We do not pass your name or contact details to these services." },
      ] },
      { h: "Storage and Services", blocks: [
        { p: "Authentication works through Google Firebase Authentication. The Website is protected by Cloudflare. Orders are processed manually. Data is stored on a secure server only for as long as needed to fulfill the order and maintain history." },
      ] },
      { h: "Cookies and Analytics", blocks: [
        { p: "We use Cloudflare's privacy-friendly analytics with no third-party advertising cookies. Cookies are used only to make account sign-in work." },
      ] },
      { h: "Your Rights", blocks: [
        { p: "You can ask us to delete your account and the associated data. Write to us at {email}." },
      ] },
      { h: "Contact", blocks: [
        { p: "For privacy questions: {email}." },
      ] },
    ],
  },

  terms: {
    title: "Terms of Use",
    sections: [
      { h: "About the Service", blocks: [
        { p: "{storeName} lets you create a 3D model of a city area or a map keychain based on open OpenStreetMap data and download a ready-to-print file (3MF/STL) or order a print." },
      ] },
      { h: "Account and Free Downloads", blocks: [
        { p: "An account is required to download the full model. Each user gets 5 free downloads. Beyond that — by arrangement (printing/payment), get in touch via the website." },
      ] },
      { h: "Data and Copyright", blocks: [
        { p: "Cartographic data © OpenStreetMap contributors (ODbL). You may use the generated files for personal printing. Reselling the service or large-scale commercial use requires a separate arrangement." },
      ] },
      { h: "Orders and Payment", blocks: [
        { p: "Orders are placed through the website; payment is made online via LiqPay or by arrangement. Price and delivery details are on the [delivery:\"Payment and Delivery\"] page, and the full terms are in the [offer:Public Offer Agreement]." },
      ] },
      { h: "Liability", blocks: [
        { p: "The service is provided \"as is\". We strive for maximum model accuracy but do not guarantee full correspondence to real-world objects due to the limitations of the source OSM data." },
      ] },
      { h: "Contact", blocks: [
        { p: "Questions: {email}." },
      ] },
    ],
  },
};
