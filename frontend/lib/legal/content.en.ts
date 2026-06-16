import type { LegalSet } from "./content";

export const en: LegalSet = {
  offer: {
    title: "Public Offer Agreement",
    intro: [
      "This document is an official public proposal (offer) by {ownerFull} (hereinafter the \"Seller\") to conclude an agreement for the sale of goods and provision of services on the terms set out below. By paying for an order on the {domain} website (hereinafter the \"Website\"), the Buyer confirms that they have fully read, understood and unconditionally accepted the terms of this Agreement (acceptance of the offer pursuant to Articles 633, 641, 642 of the Civil Code of Ukraine).",
    ],
    sections: [
      { h: "1. Definitions", blocks: [
        { p: "Seller — {ownerShort}, Tax ID {taxId}." },
        { p: "Buyer — any legally capable person who has placed an order on the Website." },
        { p: "Goods / Service — a digital 3D model (a 3MF/STL file) of a city map or a map keychain, and/or the manufacture (3D printing) of a physical product from Eco PLA bioplastic made to the Buyer's individual order." },
      ] },
      { h: "2. Subject of the Agreement", blocks: [
        { p: "The Seller undertakes to provide the Buyer with a digital 3D model and/or to manufacture and deliver a physical product to order, and the Buyer undertakes to accept and pay for them in accordance with the terms of this Agreement." },
        { p: "Each product is manufactured individually according to parameters (location on the map, shape, size, text) that the Buyer selects themselves in the Website's designer, and is therefore goods made to order." },
      ] },
      { h: "3. Placing an Order", blocks: [
        { p: "The Buyer creates an order in the designer on the Website and provides contact details (name, phone, delivery method and address). The order is deemed accepted once it has been paid for or confirmed by an operator." },
        { p: "The Buyer is responsible for the accuracy of the data provided. The Seller bears no responsibility for the consequences of errors in the data supplied by the Buyer." },
      ] },
      { h: "4. Price and Payment", blocks: [
        { p: "Prices for goods and services are shown on the Website in hryvnia (for orders within Ukraine) and in euros (for delivery to the EU) and are indicative until the order is confirmed. The Buyer sees the final cost at the checkout step." },
        { p: "Payment is made online through the LiqPay payment service (by Visa/Mastercard bank card) or by another agreed method. Downloading the finished digital file within the free limit is free of charge." },
        { p: "Details are available on the [delivery:\"Payment and Delivery\"] page." },
      ] },
      { h: "5. Manufacture and Delivery", blocks: [
        { p: "The digital file is provided in your account / by email immediately or after the order is confirmed. The physical product is manufactured and dispatched within the time stated on the \"Payment and Delivery\" page, via Nova Poshta or Ukrposhta (Ukraine), Nova Post EU or Meest (EU)." },
      ] },
      { h: "6. Refunds", blocks: [
        { p: "Since the goods are manufactured to individual order and the digital files are in the nature of electronic content, refunds are governed by a separate document — [refund:\"Refunds and Exchanges\"] — which forms an integral part of this Agreement." },
      ] },
      { h: "7. Intellectual Property Rights", blocks: [
        { p: "The cartographic data is provided by OpenStreetMap (ODbL), and the elevation data comes from open sources. The generated 3D model is provided to the Buyer for personal, non-commercial use and printing. A GPX route uploaded by the Buyer is their own data and is processed solely to build the model (see the [privacy:Privacy Policy])." },
      ] },
      { h: "8. Liability of the Parties", blocks: [
        { p: "The Seller is not responsible for print quality on the Buyer's own equipment when the Buyer prints the downloaded file themselves. The Seller's aggregate liability is limited to the amount paid for the order." },
        { p: "The parties are released from liability for failure to perform their obligations as a result of force majeure circumstances." },
      ] },
      { h: "9. Personal Data", blocks: [
        { p: "By placing an order, the Buyer consents to the processing of their personal data for the purpose of fulfilling the order in accordance with Ukraine's Law on the Protection of Personal Data and the [privacy:Privacy Policy]." },
      ] },
      { h: "10. Dispute Resolution and Term", blocks: [
        { p: "Disputes are resolved through negotiation and, failing agreement, in accordance with the applicable laws of Ukraine. The Agreement is effective from the moment of acceptance until the parties have fully performed their obligations. The Seller has the right to amend the terms by publishing a new version on this page." },
      ] },
      { h: "11. Seller's Details", blocks: [
        { kv: [
          { k: "Seller", v: "{ownerFull}" },
          { k: "Tax ID", v: "{taxId}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Registered address", v: "{ownerRegAddress}" },
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
          "3D district map: S 5.5 cm — 150 ₴, M 8 cm — 240 ₴, L 11 cm — 360 ₴, XL 15 cm — 550 ₴ (maps — from 4 €).",
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
        { p: "European Union (15 countries):" },
        { ul: ["Nova Post EU.", "Meest."] },
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
          { k: "Registered address", v: "{ownerRegAddress}" },
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
