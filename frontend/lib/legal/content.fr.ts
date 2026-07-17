import type { LegalSet } from "./content";

export const fr: LegalSet = {
  offer: {
    title: "Contrat d'offre publique",
    intro: [
      "Ce document constitue une proposition publique officielle (offre) de {ownerFull} (ci-après le « Vendeur ») de conclure un contrat de vente de marchandises et de prestation de services aux conditions exposées ci-dessous. En payant ta commande sur le site {domain} (ci-après le « Site »), l'Acheteur confirme qu'il a entièrement lu, compris et accepté sans réserve les conditions du présent Contrat (acceptation de l'offre conformément aux art. 633, 641, 642 du Code civil de l'Ukraine).",
    ],
    sections: [
      { h: "1. Définitions", blocks: [
        { p: "Vendeur — {ownerShort}, numéro fiscal {taxId}." },
        { p: "Acheteur — toute personne capable ayant passé une commande sur le Site." },
        { p: "Marchandise / Service — modèle 3D numérique (fichier 3MF/STL) d'une carte de ville ou d'un porte-clés-carte et/ou fabrication (impression 3D) d'un produit physique en bioplastique Eco PLA sur commande individuelle de l'Acheteur." },
      ] },
      { h: "2. Objet du contrat", blocks: [
        { p: "Le Vendeur s'engage à fournir à l'Acheteur un modèle 3D numérique et/ou à fabriquer et remettre un produit physique sur commande, et l'Acheteur — à l'accepter et à le payer conformément aux conditions du présent Contrat." },
        { p: "Chaque produit est fabriqué individuellement selon les paramètres (lieu sur la carte, forme, taille, texte) que l'Acheteur choisit lui-même dans le configurateur du Site, c'est-à-dire qu'il s'agit d'un bien fabriqué sur commande." },
      ] },
      { h: "3. Passation de la commande", blocks: [
        { p: "L'Acheteur constitue sa commande dans le configurateur du Site et indique ses coordonnées (nom, téléphone, mode et adresse de livraison). La commande est réputée acceptée après son paiement ou sa confirmation par un opérateur." },
        { p: "L'Acheteur est responsable de l'exactitude des données fournies. Le Vendeur n'est pas responsable des conséquences des erreurs dans les données fournies par l'Acheteur." },
      ] },
      { h: "4. Prix et paiement", blocks: [
        { p: "Les prix des marchandises et des services sont indiqués sur le Site en hryvnias (pour les commandes en Ukraine) et en euros (pour la livraison dans l'UE) et sont indicatifs jusqu'à la confirmation de la commande. L'Acheteur voit le prix final à l'étape de validation." },
        { p: "Le paiement s'effectue en ligne via le service de paiement LiqPay (par carte bancaire Visa/Mastercard) ou par tout autre moyen convenu. Le téléchargement du fichier numérique fini dans la limite gratuite est sans frais." },
        { p: "Détails — sur la page [delivery:« Paiement et livraison »]." },
      ] },
      { h: "5. Fabrication et livraison", blocks: [
        { p: "Le fichier numérique est fourni dans l'espace client / par e-mail immédiatement ou après la confirmation de la commande. Le produit physique est fabriqué et expédié dans le délai indiqué sur la page « Paiement et livraison », par les services Nova Poshta ou Ukrposhta (Ukraine), Nova Post EU ou Meest (UE)." },
      ] },
      { h: "6. Remboursement", blocks: [
        { p: "Étant donné que les produits sont fabriqués sur commande individuelle et que les fichiers numériques ont le caractère de contenu électronique, le remboursement est régi par un document distinct — [refund:« Retours et remboursements »], qui fait partie intégrante du présent Contrat." },
      ] },
      { h: "7. Droits de propriété intellectuelle", blocks: [
        { p: "Les données cartographiques sont fournies par OpenStreetMap (ODbL), les données d'altitude proviennent de sources ouvertes. Le modèle 3D généré est fourni à l'Acheteur pour un usage personnel non commercial et pour l'impression. L'itinéraire GPX téléchargé par l'Acheteur constitue ses propres données et est traité exclusivement pour la construction du modèle (voir la [privacy:Politique de confidentialité])." },
      ] },
      { h: "8. Responsabilité des parties", blocks: [
        { p: "Le Vendeur n'est pas responsable de la qualité de l'impression sur l'équipement de l'Acheteur en cas d'impression autonome du fichier téléchargé. La responsabilité totale du Vendeur est limitée au montant de la commande payée." },
        { p: "Les parties sont dégagées de toute responsabilité en cas d'inexécution de leurs obligations résultant de circonstances de force majeure." },
      ] },
      { h: "9. Données personnelles", blocks: [
        { p: "En passant commande, l'Acheteur consent au traitement de ses données personnelles aux fins de l'exécution de la commande, conformément à la loi ukrainienne sur la protection des données personnelles et à la [privacy:Politique de confidentialité]." },
      ] },
      { h: "10. Règlement des litiges et durée de validité", blocks: [
        { p: "Les litiges sont réglés par voie de négociation et, à défaut d'accord — conformément à la législation en vigueur de l'Ukraine. Le Contrat prend effet dès l'acceptation et jusqu'à l'exécution complète des obligations par les parties. Le Vendeur a le droit de modifier les conditions en publiant une nouvelle version sur cette page." },
      ] },
      { h: "11. Coordonnées du Vendeur", blocks: [
        { kv: [
          { k: "Nom", v: "{ownerFull}" },
          { k: "Numéro fiscal (IPN/RNOKPP)", v: "{taxId}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Magasin", v: "{storeName}, {storeAddress}" },
          { k: "E-mail", v: "{email}" },
          { k: "Téléphone", v: "{phone}" },
        ] },
      ] },
    ],
  },

  refund: {
    title: "Retours et remboursements",
    sections: [
      { h: "Nature du produit", blocks: [
        { p: "Tous les produits {storeName} sont fabriqués individuellement selon les paramètres que l'Acheteur choisit lui-même (lieu sur la carte, forme, taille, texte), et les modèles 3D numériques constituent du contenu électronique. Cela influe sur les conditions de retour conformément à la loi ukrainienne sur la protection des droits des consommateurs (un bien de qualité conforme, fabriqué sur commande individuelle, ne peut être ni retourné ni échangé)." },
      ] },
      { h: "Fichiers numériques (téléchargement 3MF/STL)", blocks: [
        { p: "Les sommes versées pour un fichier numérique ne sont pas remboursées une fois que le fichier a été généré et mis à disposition pour le téléchargement, car le service a déjà été pleinement rendu. Si le fichier ne se télécharge pas techniquement ou est endommagé de notre fait — nous le régénérons gratuitement ou te remboursons." },
      ] },
      { h: "Impression sur commande (produit physique)", blocks: [
        { ul: [
          "Avant le lancement en production — tu peux annuler la commande et obtenir un remboursement intégral si nous n'avons pas encore commencé l'impression.",
          "Après le lancement en production — les sommes ne sont pas remboursées, car le produit est fabriqué personnellement pour ta commande.",
          "Défaut, dommage lors de la livraison ou non-conformité à la commande — nous réimprimons et réexpédions le produit gratuitement ou remboursons l'intégralité du prix (à ton choix).",
        ] },
      ] },
      { h: "Comment effectuer un retour", blocks: [
        { p: "Écris à {email} ou appelle le {phone}, en indiquant le numéro de commande, le motif et (en cas de défaut) une photo du produit. Nous examinerons ta demande sous 1–3 jours ouvrés." },
        { p: "Le remboursement s'effectue par le même moyen que celui utilisé pour le paiement (remboursement sur la carte via LiqPay), dans les délais prévus par les règles du système de paiement et de la banque." },
      ] },
      { h: "Contact", blocks: [
        { p: "Ce document fait partie intégrante du [offer:Contrat d'offre publique]. Questions : {email}." },
      ] },
    ],
  },

  delivery: {
    title: "Paiement et livraison",
    sections: [
      { h: "Produits et prix", blocks: [
        { p: "Prix des produits (la livraison est facturée séparément selon le tarif du transporteur) :" },
        { ul: [
          "Porte-clés-carte — à partir de 120 ₴ (≈ 3 €).",
          "Carte 3D d'un quartier : S 5,5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴ (cartes — à partir de 6 €).",
          "Magnet de réfrigérateur (carte) — 150 ₴.",
          "Téléchargement du fichier fini 3MF/STL pour une impression autonome — gratuit dans la limite du compte.",
        ] },
      ] },
      { h: "Paiement", blocks: [
        { p: "Paiement en ligne par carte bancaire Visa / Mastercard via le service sécurisé LiqPay. Les données de la carte sont traitées du côté du système de paiement — nous ne les conservons pas. Le paiement par accord avec un opérateur est également possible. Le fichier numérique dans la limite gratuite est fourni sans paiement." },
      ] },
      { h: "Fabrication", blocks: [
        { p: "Les produits sont imprimés sur commande en bioplastique Eco PLA. Délai de fabrication indicatif — 1–3 jours ouvrés après la confirmation de la commande (selon la charge et la complexité)." },
      ] },
      { h: "Livraison", blocks: [
        { p: "Ukraine :" },
        { ul: ["Nova Poshta — point relais ou casier automatique.", "Ukrposhta — bureau de poste."] },
        { p: "Union européenne (15 pays) :" },
        { ul: ["Nova Post EU.", "Meest."] },
        { p: "Les frais de livraison sont calculés selon les tarifs du transporteur et payés séparément (généralement à la réception). Délai de livraison indicatif en Ukraine — 2–4 jours ouvrés après l'expédition." },
      ] },
      { h: "Retours", blocks: [
        { p: "Les conditions de remboursement sont décrites sur la page [refund:« Retours et remboursements »]. Les conditions générales — dans le [offer:Contrat d'offre publique]." },
      ] },
      { h: "Contact", blocks: [
        { p: "Questions relatives au paiement ou à la livraison : {email}, {phone}." },
      ] },
    ],
  },

  contacts: {
    title: "Contacts et coordonnées",
    sections: [
      { h: "Nous contacter", blocks: [
        { kv: [
          { k: "E-mail", v: "{email}" },
          { k: "Téléphone", v: "{phone}" },
          { k: "Site web", v: "{domain}" },
        ] },
        { p: "Horaires de traitement des commandes : tous les jours, nous répondons sous 24 heures." },
      ] },
      { h: "Magasin", blocks: [
        { kv: [
          { k: "Nom", v: "{storeName}" },
          { k: "Adresse du magasin", v: "{storeAddress}" },
        ] },
      ] },
      { h: "Vendeur (entrepreneur individuel)", blocks: [
        { kv: [
          { k: "Nom", v: "{ownerFull}" },
          { k: "Numéro fiscal (IPN/RNOKPP)", v: "{taxId}" },
          { k: "Activité (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
        ] },
      ] },
      { h: "Ce que nous vendons", blocks: [
        { p: "{storeName} — ce sont des modèles 3D de cartes de villes et de porte-clés-cartes sur commande. Tu peux télécharger un fichier prêt à imprimer (3MF/STL) dans la limite gratuite ou commander l'impression d'un produit en bioplastique Eco PLA avec livraison. Les prix indicatifs sont indiqués dans le configurateur et sur la page [delivery:« Paiement et livraison »]." },
      ] },
      { h: "Documents", blocks: [
        { ul: [
          "[offer:Contrat d'offre publique]",
          "[refund:Retours et remboursements]",
          "[delivery:Paiement et livraison]",
          "[privacy:Politique de confidentialité]",
          "[terms:Conditions d'utilisation]",
        ] },
      ] },
    ],
  },

  privacy: {
    title: "Politique de confidentialité",
    sections: [
      { h: "Quelles données nous collectons", blocks: [
        { p: "Le nom, l'e-mail ou le numéro de téléphone (lors de la connexion / l'inscription), ainsi que les coordonnées et l'adresse de livraison que tu indiques lors de la passation de la commande. Les données techniques (l'historique des modèles générés) sont conservées dans ton espace client." },
      ] },
      { h: "Comment nous utilisons les données", blocks: [
        { p: "Exclusivement pour la fourniture du service : la connexion au compte, la sauvegarde de l'historique des modèles, le traitement et la livraison des commandes, le contact avec toi à propos de la commande. Nous ne vendons ni ne transmettons tes données à des tiers à des fins publicitaires." },
      ] },
      { h: "Itinéraires téléchargés (GPX) et géodonnées", blocks: [
        { p: "Si tu téléverses un fichier GPX (par exemple l'export de ta propre activité depuis Strava ou une autre application), nous traitons les coordonnées de l'itinéraire exclusivement pour la construction de ton modèle 3D. Ce sont tes propres données — nous ne les publions pas, ne les transmettons pas à des tiers et ne les utilisons pas à des fins publicitaires. Les points de l'itinéraire sont simplifiés et conservés exactement le temps nécessaire à la génération et (si tu es connecté) à la tenue de l'historique des modèles dans ton espace client ; tu peux demander leur suppression à tout moment." },
        { p: "La recherche d'un lieu sur la carte envoie ta requête au service de géocodage d'OpenStreetMap (Nominatim), et les cartes elles-mêmes sont chargées depuis les tuiles d'OpenStreetMap — conformément à leurs conditions d'utilisation. Nous ne transmettons à ces services ni ton nom ni tes coordonnées." },
      ] },
      { h: "Conservation et services", blocks: [
        { p: "L'authentification fonctionne via Google Firebase Authentication. Le Site est protégé par Cloudflare. Les commandes sont traitées manuellement. Les données sont conservées sur un serveur sécurisé exactement le temps nécessaire à l'exécution de la commande et à la tenue de l'historique." },
      ] },
      { h: "Cookies et analytique", blocks: [
        { p: "Nous utilisons l'analytique privée de Cloudflare, sans cookies publicitaires tiers. Les cookies ne servent qu'au fonctionnement de la connexion au compte." },
      ] },
      { h: "Tes droits", blocks: [
        { p: "Tu peux demander la suppression de ton compte et des données associées. Écris-nous à {email}." },
      ] },
      { h: "Contact", blocks: [
        { p: "Pour les questions de confidentialité : {email}." },
      ] },
    ],
  },

  terms: {
    title: "Conditions d'utilisation",
    sections: [
      { h: "À propos du service", blocks: [
        { p: "{storeName} te permet de créer un modèle 3D d'une zone urbaine ou d'un porte-clés avec une carte à partir des données ouvertes d'OpenStreetMap et de télécharger un fichier prêt pour l'impression 3D (3MF/STL) ou de commander une impression." },
      ] },
      { h: "Compte et téléchargements gratuits", blocks: [
        { p: "Le téléchargement du modèle complet nécessite un compte. Chaque utilisateur dispose de 5 téléchargements gratuits. Au-delà — par accord (impression / paiement), contact via le site." },
      ] },
      { h: "Données et droits d'auteur", blocks: [
        { p: "Données cartographiques © OpenStreetMap contributors (ODbL). Tu peux utiliser les fichiers générés pour ton impression personnelle. La revente du service ou un usage commercial à grande échelle nécessite un accord distinct." },
      ] },
      { h: "Commande et paiement", blocks: [
        { p: "La commande est passée via le site ; le paiement — en ligne via LiqPay ou par accord. Les détails des prix et de la livraison — sur la page [delivery:« Paiement et livraison »], les conditions complètes — dans le [offer:Contrat d'offre publique]." },
      ] },
      { h: "Responsabilité", blocks: [
        { p: "Le service est fourni « en l'état ». Nous visons une précision maximale des modèles, mais ne garantissons pas une correspondance totale avec les objets réels en raison des limites des données sources OSM." },
      ] },
      { h: "Contact", blocks: [
        { p: "Questions : {email}." },
      ] },
    ],
  },
};
