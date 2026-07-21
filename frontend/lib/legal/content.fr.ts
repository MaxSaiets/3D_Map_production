import type { LegalSet } from "./content";

export const fr: LegalSet = {
  offer: {
    title: "Contrat d'offre publique",
    intro: [
      "Le présent document constitue la proposition publique officielle (offre) de {ownerFull} (ci-après le « Vendeur ») de conclure un contrat de vente de biens et de prestation de services à distance aux conditions exposées ci-dessous, conformément aux articles 633, 641 et 642 du Code civil d'Ukraine et aux lois d'Ukraine « Sur le commerce électronique » et « Sur la protection des droits des consommateurs ».",
      "En passant et/ou en payant une commande sur le site {domain} (ci-après le « Site »), l'Acheteur confirme avoir lu intégralement, compris et accepté sans réserve les conditions du présent Contrat (acceptation de l'offre). L'acceptation de la présente offre équivaut à la conclusion d'un contrat par écrit.",
    ],
    sections: [
      { h: "1. Termes et définitions", blocks: [
        { kv: [
          { k: "Vendeur", v: "{ownerFull}, numéro fiscal (RNOKPP) {taxId}, assujetti à l'impôt unique." },
          { k: "Acheteur", v: "toute personne physique ou morale dotée de la capacité juridique ayant passé une commande sur le Site et accepté la présente offre." },
          { k: "Site", v: "la boutique en ligne {storeName} à l'adresse {domain}, y compris le configurateur en ligne de modèles 3D." },
          { k: "Configurateur", v: "le service logiciel du Site dans lequel l'Acheteur choisit lui-même les paramètres du futur produit : zone de la carte, forme, taille, format, texte de gravure, etc." },
          { k: "Produit", v: "un article physique (carte de ville imprimée en 3D, panneau, porte-clés carte, aimant, etc.) en bioplastique Eco PLA, fabriqué sur commande individuelle de l'Acheteur." },
          { k: "Contenu numérique", v: "un modèle 3D numérique généré (fichier au format 3MF/STL), adapté à l'impression 3D autonome." },
          { k: "Commande", v: "la demande de l'Acheteur, passée via le Site, d'acquérir le Produit et/ou le Contenu numérique." },
        ] },
      ] },
      { h: "2. Objet du contrat", blocks: [
        { p: "Le Vendeur s'engage, à la demande de l'Acheteur, à générer un modèle 3D numérique et/ou à fabriquer un article physique selon les paramètres individuels de l'Acheteur et à le lui remettre, et l'Acheteur s'engage à accepter et à payer le Produit/Contenu numérique aux conditions du présent Contrat." },
        { p: "Chaque article est fabriqué individuellement selon les paramètres (emplacement sur la carte, forme, taille, texte, couleurs) que l'Acheteur choisit lui-même dans le Configurateur. Par conséquent, le Produit est un article fabriqué sur commande individuelle de l'Acheteur au sens de la loi d'Ukraine « Sur la protection des droits des consommateurs »." },
        { p: "L'assortiment, les caractéristiques et les prix indicatifs des Produits figurent sur le Site. Les photographies et les aperçus 3D ont un caractère informatif : l'article fini peut présenter de légères différences de teinte du matériau et de texture des couches, ce qui constitue une particularité naturelle de la technologie d'impression 3D et non un défaut du Produit." },
      ] },
      { h: "3. Passation de la commande", blocks: [
        { ul: [
          "L'Acheteur crée lui-même le modèle dans le Configurateur : il choisit la ville/zone de la carte, la forme, la taille, le format (en relief/plat, aimant) et, s'il le souhaite, un texte de gravure ou un itinéraire GPX.",
          "Avant de finaliser la commande, l'Acheteur voit un aperçu 3D du modèle et le coût définitif de la commande.",
          "Pour passer commande, l'Acheteur indique ses coordonnées : nom, téléphone, e-mail si nécessaire, mode de livraison et adresse/agence du transporteur.",
          "La commande est réputée acceptée pour exécution après son paiement en ligne ou après confirmation par l'opérateur (selon accord).",
        ] },
        { p: "L'Acheteur est seul responsable de l'exactitude des paramètres du modèle choisis (zone de la carte, texte, taille) et de la véracité de ses coordonnées. Le Vendeur décline toute responsabilité quant aux conséquences des erreurs dans les données fournies par l'Acheteur, notamment la fabrication d'un article avec un texte ou une zone de carte erronés que l'Acheteur a lui-même validés dans le Configurateur." },
      ] },
      { h: "4. Prix et modalités de paiement", blocks: [
        { p: "Les prix sur le Site sont indiqués en hryvnias (pour les commandes en Ukraine) et en euros (à titre indicatif, pour la livraison vers les pays de l'UE). L'Acheteur voit le coût définitif de la commande à l'étape de finalisation, avant le paiement. Les frais de livraison ne sont pas inclus dans le prix du Produit et sont payés séparément selon les tarifs du transporteur." },
        { p: "Le paiement s'effectue en ligne via le service de paiement LiqPay (JSC CB « PrivatBank ») : par carte bancaire Visa/Mastercard et par d'autres moyens disponibles dans LiqPay. Les données de la carte de paiement sont traitées du côté du système de paiement ; le Vendeur ne les reçoit pas et ne les conserve pas." },
        { p: "Sur accord avec l'opérateur, un autre mode de paiement convenu est possible. Le téléchargement du Contenu numérique fini dans la limite gratuite du compte est gratuit." },
        { p: "Le Produit est payé intégralement avant sa mise en production, sauf accord contraire des parties. Détails sur la page [delivery:« Paiement et livraison »]." },
      ] },
      { h: "5. Délais de fabrication", blocks: [
        { p: "Le Contenu numérique est généré automatiquement et mis à disposition dans le compte de l'Acheteur / par e-mail immédiatement après la génération ou après confirmation de la commande." },
        { p: "Le délai indicatif de fabrication d'un article physique est de 1 à 3 jours ouvrables à compter du paiement/de la confirmation de la commande. En cas de forte charge de travail ou de complexité technique de l'article, le délai peut être prolongé, ce dont le Vendeur informe l'Acheteur." },
      ] },
      { h: "6. Livraison", blocks: [
        { p: "La livraison en Ukraine est assurée par les services « Nova Poshta » (agence, consigne automatique) ou « Ukrposhta ». La livraison vers les pays de l'UE — par Nova Post EU ou Meest. Le délai indicatif de livraison en Ukraine est de 2 à 4 jours ouvrables après l'expédition." },
        { p: "Les frais de livraison sont calculés selon les tarifs du transporteur et payés séparément par l'Acheteur (généralement à la réception). Le droit de propriété sur le Produit et le risque de détérioration fortuite sont transférés à l'Acheteur au moment de la réception du Produit auprès du transporteur." },
        { p: "À la réception, l'Acheteur est tenu d'inspecter l'envoi afin de vérifier l'intégrité de l'emballage et de l'article. En cas de dommage pendant le transport, il convient de le faire constater par un procès-verbal du transporteur et d'en informer le Vendeur — un tel cas est résolu par une réimpression gratuite ou un remboursement (voir la section 8)." },
      ] },
      { h: "7. Qualité et garantie", blocks: [
        { p: "Le Vendeur garantit la conformité de l'article aux paramètres validés par l'Acheteur dans le Configurateur ainsi qu'une qualité d'impression appropriée. Les articles physiques bénéficient d'une garantie de 60 jours à compter de la réception couvrant les défauts d'impression et le délaminage non imputables à l'Acheteur." },
        { p: "Les particularités naturelles de la technologie d'impression FDM (couches d'impression visibles, légères différences de teinte du plastique entre les lots, traces technologiques sur la surface inférieure) ne constituent pas des défauts du Produit." },
        { p: "Le Vendeur n'est pas responsable du résultat de l'impression sur l'équipement de l'Acheteur lorsque celui-ci imprime lui-même le Contenu numérique téléchargé (la qualité d'une telle impression dépend de l'imprimante, du matériau et des réglages de l'Acheteur)." },
      ] },
      { h: "8. Remboursement et échange", blocks: [
        { p: "Le Produit étant fabriqué sur commande individuelle selon les paramètres uniques de l'Acheteur, le Produit de qualité conforme n'est pas soumis au retour ni à l'échange (loi d'Ukraine « Sur la protection des droits des consommateurs » ; liste de produits approuvée par la résolution du Cabinet des ministres d'Ukraine n° 172 du 19.03.1994). Une fois l'accès au téléchargement fourni, le Contenu numérique n'est pas remboursable en tant que contenu électronique dont le service a été consommé." },
        { ul: [
          "Avant la mise en production de la commande, l'Acheteur peut l'annuler et obtenir un remboursement intégral.",
          "En cas de défaut, de dommage pendant la livraison ou de non-conformité de l'article aux paramètres validés, le Vendeur, au choix de l'Acheteur, fabrique et expédie gratuitement un nouvel article ou rembourse le prix intégral.",
          "Si le Contenu numérique est techniquement endommagé ou ne se télécharge pas par la faute du Vendeur, le fichier est régénéré gratuitement ou les fonds sont remboursés.",
        ] },
        { p: "La procédure de réclamation et les délais sont décrits dans le document [refund:« Retours et remboursements »], qui fait partie intégrante du présent Contrat. Le remboursement est effectué par le même moyen que le paiement (sur la carte via LiqPay), dans les délais prévus par les règles du système de paiement et de la banque." },
      ] },
      { h: "9. Droits et obligations des parties", blocks: [
        { p: "Le Vendeur est tenu de : fabriquer le Produit conformément aux paramètres validés par l'Acheteur ; respecter les délais annoncés ; informer l'Acheteur de l'état de la commande ; assurer la confidentialité des données personnelles de l'Acheteur." },
        { p: "Le Vendeur a le droit de : faire appel à des tiers pour l'exécution de ses obligations (transporteurs, services de paiement) ; suspendre l'exécution de la commande en cas de non-paiement ; refuser de fabriquer un article dont le contenu enfreint la législation ukrainienne (notamment s'il contient des symboles interdits ou des discours de haine), avec remboursement intégral." },
        { p: "L'Acheteur est tenu de : fournir des données exactes pour l'exécution de la commande ; payer la commande ; retirer le Produit auprès du transporteur dans les délais de conservation de l'envoi." },
        { p: "L'Acheteur a le droit de : recevoir un Produit de qualité conforme dans le délai annoncé ; obtenir des informations sur l'état de sa commande ; déposer une réclamation selon la procédure prévue par le présent Contrat." },
      ] },
      { h: "10. Droits de propriété intellectuelle", blocks: [
        { p: "Les données cartographiques sont fournies par OpenStreetMap (© OpenStreetMap contributors, licence ODbL) ; les données d'altitude proviennent de sources ouvertes. Le modèle 3D généré est fourni à l'Acheteur pour un usage personnel non commercial et pour impression. L'utilisation commerciale à grande échelle ou la revente des modèles nécessite un accord écrit distinct avec le Vendeur." },
        { p: "L'itinéraire GPX téléversé par l'Acheteur constitue ses propres données et n'est traité que pour construire son modèle (voir la [privacy:Politique de confidentialité]). L'Acheteur garantit que le texte de gravure qu'il commande ne porte pas atteinte aux droits de tiers." },
      ] },
      { h: "11. Données personnelles", blocks: [
        { p: "En passant commande, l'Acheteur consent au traitement de ses données personnelles (nom, coordonnées, adresse de livraison) exclusivement aux fins de l'exécution du présent Contrat, conformément à la loi d'Ukraine « Sur la protection des données personnelles » et à la [privacy:Politique de confidentialité]. Les données ne sont pas transmises à des tiers, sauf lorsque cela est nécessaire à l'exécution de la commande (transporteur, service de paiement)." },
      ] },
      { h: "12. Responsabilité et force majeure", blocks: [
        { p: "En cas d'inexécution ou de mauvaise exécution de leurs obligations, les parties engagent leur responsabilité conformément à la législation ukrainienne en vigueur. La responsabilité totale du Vendeur au titre de toute réclamation est limitée au montant de la commande effectivement payé par l'Acheteur." },
        { p: "Les parties sont exonérées de toute responsabilité en cas d'inexécution totale ou partielle de leurs obligations si celle-ci résulte de circonstances de force majeure : actions militaires, bombardements, coupures d'électricité, catastrophes naturelles, décisions des autorités, défaillances des transporteurs, etc. Les délais d'exécution sont prolongés de la durée de ces circonstances." },
      ] },
      { h: "13. Réclamations et règlement des litiges", blocks: [
        { p: "Les réclamations concernant une commande sont acceptées à {email} ou par téléphone au {phone}, en indiquant le numéro de commande. Le Vendeur examine les demandes sous 1 à 3 jours ouvrables. Les litiges sont réglés par voie de négociation et, à défaut d'accord, selon la procédure établie par la législation ukrainienne en vigueur." },
      ] },
      { h: "14. Durée et modification des conditions", blocks: [
        { p: "Le Contrat entre en vigueur au moment de l'acceptation de l'offre par l'Acheteur et reste en vigueur jusqu'à la pleine exécution des obligations par les parties. Le Vendeur a le droit de modifier les conditions de la présente offre en publiant une nouvelle version sur cette page ; la nouvelle version s'applique aux commandes passées après sa publication. La version en vigueur est disponible en permanence à l'adresse {domain}/offer." },
        { p: "Font partie intégrante du présent Contrat les documents : [refund:« Retours et remboursements »], [delivery:« Paiement et livraison »], [privacy:« Politique de confidentialité »] et [terms:« Conditions d'utilisation »]." },
      ] },
      { h: "15. Coordonnées du Vendeur", blocks: [
        { kv: [
          { k: "Vendeur", v: "{ownerFull}" },
          { k: "Numéro fiscal (RNOKPP)", v: "{taxId}" },
          { k: "Type d'activité (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Boutique", v: "{storeName}, {storeAddress}" },
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
