import type { LegalSet } from "./content";

export const es: LegalSet = {
  offer: {
    title: "Contrato de oferta pública",
    intro: [
      "El presente documento constituye la propuesta pública oficial (oferta) de {ownerFull} (en adelante, el «Vendedor») para celebrar un contrato de compraventa de bienes y prestación de servicios a distancia en las condiciones expuestas a continuación, de conformidad con los artículos 633, 641 y 642 del Código Civil de Ucrania y las Leyes de Ucrania «Sobre el comercio electrónico» y «Sobre la protección de los derechos de los consumidores».",
      "Al realizar y/o pagar un pedido en el sitio web {domain} (en adelante, el «Sitio»), el Comprador confirma que ha leído íntegramente, comprendido y aceptado sin reservas las condiciones del presente Contrato (aceptación de la oferta). La aceptación de esta oferta equivale a la celebración de un contrato por escrito.",
    ],
    sections: [
      { h: "1. Términos y definiciones", blocks: [
        { kv: [
          { k: "Vendedor", v: "{ownerFull}, número fiscal (RNOKPP) {taxId}, contribuyente del impuesto único." },
          { k: "Comprador", v: "cualquier persona física o jurídica con capacidad legal que haya realizado un pedido en el Sitio y aceptado esta oferta." },
          { k: "Sitio", v: "la tienda en línea {storeName} en la dirección {domain}, incluido el configurador en línea de modelos 3D." },
          { k: "Configurador", v: "el servicio de software del Sitio en el que el Comprador elige por sí mismo los parámetros del futuro producto: zona del mapa, forma, tamaño, formato, texto del grabado, etc." },
          { k: "Producto", v: "un artículo físico (mapa de ciudad impreso en 3D, panel, llavero-mapa, imán, etc.) de bioplástico Eco PLA, fabricado por encargo individual del Comprador." },
          { k: "Contenido digital", v: "un modelo 3D digital generado (archivo en formato 3MF/STL), apto para la impresión 3D por cuenta propia." },
          { k: "Pedido", v: "la solicitud del Comprador, realizada a través del Sitio, para adquirir el Producto y/o el Contenido digital." },
        ] },
      ] },
      { h: "2. Objeto del contrato", blocks: [
        { p: "El Vendedor se compromete, por encargo del Comprador, a generar un modelo 3D digital y/o fabricar un artículo físico según los parámetros individuales del Comprador y entregárselo, y el Comprador se compromete a aceptar y pagar el Producto/Contenido digital en las condiciones del presente Contrato." },
        { p: "Cada artículo se fabrica individualmente según los parámetros (lugar en el mapa, forma, tamaño, texto, colores) que el Comprador elige por sí mismo en el Configurador. En consecuencia, el Producto es un artículo fabricado por encargo individual del Comprador en el sentido de la Ley de Ucrania «Sobre la protección de los derechos de los consumidores»." },
        { p: "El surtido, las características y los precios orientativos de los Productos figuran en el Sitio. Las fotografías y las vistas previas 3D tienen carácter informativo: el artículo terminado puede presentar ligeras diferencias en el tono del material y la textura de las capas, lo cual es una particularidad natural de la tecnología de impresión 3D y no constituye un defecto del Producto." },
      ] },
      { h: "3. Procedimiento de realización del pedido", blocks: [
        { ul: [
          "El Comprador crea el modelo por sí mismo en el Configurador: elige la ciudad/zona del mapa, la forma, el tamaño, el formato (en relieve/plano, imán) y, si lo desea, el texto del grabado o una ruta GPX.",
          "Antes de formalizar el pedido, el Comprador ve una vista previa 3D del modelo y el coste final del pedido.",
          "Para formalizar el pedido, el Comprador indica sus datos de contacto: nombre, teléfono, correo electrónico si es necesario, método de entrega y dirección/sucursal del transportista.",
          "El pedido se considera aceptado para su ejecución tras su pago en línea o tras la confirmación por parte del operador (según lo acordado).",
        ] },
        { p: "El Comprador es el único responsable de la corrección de los parámetros del modelo elegidos (zona del mapa, texto, tamaño) y de la veracidad de los datos de contacto. El Vendedor no se hace responsable de las consecuencias de los errores en los datos facilitados por el Comprador, en particular de la fabricación de un artículo con un texto o una zona del mapa erróneos que el propio Comprador aprobó en el Configurador." },
      ] },
      { h: "4. Precio y forma de pago", blocks: [
        { p: "Los precios del Sitio se indican en grivnas (para pedidos dentro de Ucrania) y en euros (orientativos, para envíos a países de la UE). El Comprador ve el coste final del pedido en el paso de formalización, antes del pago. Los gastos de envío no están incluidos en el precio del Producto y se abonan por separado según las tarifas del transportista." },
        { p: "El pago se realiza en línea a través del servicio de pago LiqPay (JSC CB «PrivatBank»): con tarjeta bancaria Visa/Mastercard y otros métodos disponibles en LiqPay. Los datos de la tarjeta de pago se procesan en el lado del sistema de pago; el Vendedor no los recibe ni los almacena." },
        { p: "Previo acuerdo con el operador, es posible el pago por otro método convenido. La descarga del Contenido digital terminado dentro del límite gratuito de la cuenta es gratuita." },
        { p: "El Producto se paga íntegramente antes de su puesta en producción, salvo acuerdo en contrario de las partes. Más detalles en la página [delivery:«Pago y entrega»]." },
      ] },
      { h: "5. Plazos de fabricación", blocks: [
        { p: "El Contenido digital se genera automáticamente y se facilita en la cuenta del Comprador o por correo electrónico inmediatamente después de la generación o tras la confirmación del pedido." },
        { p: "El plazo orientativo de fabricación del artículo físico es de 1 a 3 días laborables desde el pago/confirmación del pedido. En caso de alta carga de trabajo o complejidad técnica del artículo, el plazo puede ampliarse, de lo cual el Vendedor informa al Comprador." },
      ] },
      { h: "6. Entrega", blocks: [
        { p: "La entrega dentro de Ucrania se realiza mediante los servicios «Nova Poshta» (sucursal, taquilla postal) o «Ukrposhta». La entrega a países de la UE, mediante los servicios Nova Post EU o Meest. El plazo orientativo de entrega dentro de Ucrania es de 2 a 4 días laborables tras el envío." },
        { p: "El coste de la entrega se calcula según las tarifas del transportista y lo abona el Comprador por separado (normalmente al recibir el pedido). La propiedad del Producto y el riesgo de daño fortuito se transfieren al Comprador en el momento de recibir el Producto del transportista." },
        { p: "Al recibirlo, el Comprador está obligado a inspeccionar el envío para comprobar la integridad del embalaje y del artículo. En caso de daños durante el transporte, deben documentarse mediante un acta del transportista y comunicarse al Vendedor: dicho caso se resuelve mediante una reimpresión gratuita o la devolución del dinero (véase la sección 8)." },
      ] },
      { h: "7. Calidad y garantía", blocks: [
        { p: "El Vendedor garantiza la conformidad del artículo con los parámetros aprobados por el Comprador en el Configurador y la debida calidad de impresión. Los artículos físicos cuentan con una garantía de 60 días desde su recepción frente a defectos de impresión y delaminación no imputables al Comprador." },
        { p: "Las particularidades naturales de la tecnología de impresión FDM (capas de impresión visibles, ligeras diferencias de tono del plástico entre lotes, marcas tecnológicas en la superficie inferior) no constituyen defectos del Producto." },
        { p: "El Vendedor no responde del resultado de la impresión en el equipo del Comprador cuando este imprime por su cuenta el Contenido digital descargado (la calidad de dicha impresión depende de la impresora, el material y los ajustes del Comprador)." },
      ] },
      { h: "8. Devolución del dinero y cambios", blocks: [
        { p: "Dado que el Producto se fabrica por encargo individual según los parámetros únicos del Comprador, el Producto de calidad adecuada no está sujeto a devolución ni cambio (Ley de Ucrania «Sobre la protección de los derechos de los consumidores»; lista de productos aprobada por la Resolución del Consejo de Ministros de Ucrania n.º 172 de 19.03.1994). El Contenido digital, una vez facilitado el acceso a la descarga, no está sujeto a devolución por tratarse de contenido electrónico cuyo servicio ya ha sido consumido." },
        { ul: [
          "Antes de que el pedido entre en producción, el Comprador puede cancelarlo y recibir la devolución íntegra del importe.",
          "En caso de defecto, daños durante la entrega o falta de conformidad del artículo con los parámetros aprobados, el Vendedor, a elección del Comprador, fabrica y envía gratuitamente un artículo nuevo o devuelve el importe íntegro.",
          "Si el Contenido digital está técnicamente dañado o no se descarga por causa imputable al Vendedor, el archivo se vuelve a generar gratuitamente o se devuelve el dinero.",
        ] },
        { p: "El procedimiento de reclamación y los plazos se describen en el documento [refund:«Devolución y reembolso de fondos»], que forma parte inseparable del presente Contrato. La devolución del dinero se efectúa por el mismo medio por el que se realizó el pago (a la tarjeta a través de LiqPay), en los plazos previstos por las normas del sistema de pago y del banco." },
      ] },
      { h: "9. Derechos y obligaciones de las partes", blocks: [
        { p: "El Vendedor está obligado a: fabricar el Producto conforme a los parámetros aprobados por el Comprador; cumplir los plazos declarados; informar al Comprador sobre el estado del pedido; garantizar la confidencialidad de los datos personales del Comprador." },
        { p: "El Vendedor tiene derecho a: recurrir a terceros para el cumplimiento de sus obligaciones (transportistas, servicios de pago); suspender la ejecución del pedido en caso de impago; negarse a fabricar un artículo cuyo contenido infrinja la legislación de Ucrania (en particular, que contenga simbología prohibida o discurso de odio), con devolución íntegra del importe." },
        { p: "El Comprador está obligado a: facilitar datos veraces para la ejecución del pedido; pagar el pedido; recoger el Producto del transportista dentro del plazo de almacenamiento del envío." },
        { p: "El Comprador tiene derecho a: recibir un Producto de calidad adecuada en el plazo declarado; recibir información sobre el estado de su pedido; presentar una reclamación según el procedimiento previsto en el presente Contrato." },
      ] },
      { h: "10. Derechos de propiedad intelectual", blocks: [
        { p: "Los datos cartográficos son proporcionados por OpenStreetMap (© OpenStreetMap contributors, licencia ODbL); los datos de elevación proceden de fuentes abiertas. El modelo 3D generado se facilita al Comprador para su uso personal no comercial y su impresión. El uso comercial masivo o la reventa de los modelos requiere un acuerdo escrito aparte con el Vendedor." },
        { p: "La ruta GPX cargada por el Comprador constituye datos propios del Comprador y se procesa exclusivamente para construir su modelo (véase la [privacy:Política de privacidad]). El Comprador garantiza que el texto de grabado que encarga no infringe derechos de terceros." },
      ] },
      { h: "11. Datos personales", blocks: [
        { p: "Al formalizar el pedido, el Comprador da su consentimiento al tratamiento de sus datos personales (nombre, datos de contacto, dirección de entrega) exclusivamente con el fin de ejecutar el presente Contrato, de conformidad con la Ley de Ucrania «Sobre la protección de los datos personales» y la [privacy:Política de privacidad]. Los datos no se ceden a terceros, salvo en los casos necesarios para la ejecución del pedido (transportista, servicio de pago)." },
      ] },
      { h: "12. Responsabilidad y fuerza mayor", blocks: [
        { p: "Por el incumplimiento o cumplimiento indebido de las obligaciones, las partes responden conforme a la legislación vigente de Ucrania. La responsabilidad total del Vendedor por cualquier reclamación se limita al importe del pedido efectivamente pagado por el Comprador." },
        { p: "Las partes quedan exentas de responsabilidad por el incumplimiento total o parcial de sus obligaciones si este se debe a circunstancias de fuerza mayor: acciones militares, bombardeos, cortes de electricidad, catástrofes naturales, decisiones de las autoridades, fallos de los transportistas, etc. Los plazos de cumplimiento se prorrogan mientras duren dichas circunstancias." },
      ] },
      { h: "13. Reclamaciones y resolución de disputas", blocks: [
        { p: "Las reclamaciones relativas al pedido se aceptan en {email} o por teléfono {phone}, indicando el número de pedido. El Vendedor examina las solicitudes en un plazo de 1 a 3 días laborables. Las disputas se resuelven mediante negociación y, a falta de acuerdo, según el procedimiento establecido por la legislación vigente de Ucrania." },
      ] },
      { h: "14. Vigencia y modificación de las condiciones", blocks: [
        { p: "El Contrato entra en vigor desde el momento de la aceptación de la oferta por el Comprador y permanece vigente hasta el pleno cumplimiento de las obligaciones por las partes. El Vendedor tiene derecho a modificar las condiciones de esta oferta publicando una nueva versión en esta página; la nueva versión se aplica a los pedidos realizados tras su publicación. La versión vigente está permanentemente disponible en {domain}/offer." },
        { p: "Forman parte inseparable del presente Contrato los documentos: [refund:«Devolución y reembolso de fondos»], [delivery:«Pago y entrega»], [privacy:«Política de privacidad»] y [terms:«Condiciones de uso»]." },
      ] },
      { h: "15. Datos del Vendedor", blocks: [
        { kv: [
          { k: "Vendedor", v: "{ownerFull}" },
          { k: "Número fiscal (RNOKPP)", v: "{taxId}" },
          { k: "Tipo de actividad (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Tienda", v: "{storeName}, {storeAddress}" },
          { k: "Email", v: "{email}" },
          { k: "Teléfono", v: "{phone}" },
        ] },
      ] },
    ],
  },

  refund: {
    title: "Devolución y reembolso de fondos",
    sections: [
      { h: "Naturaleza del producto", blocks: [
        { p: "Todos los productos de {storeName} se fabrican de forma individual según los parámetros que el propio Comprador elige (lugar en el mapa, forma, tamaño, texto), y los modelos 3D digitales constituyen contenido electrónico. Esto afecta a las condiciones de devolución conforme a la Ley de Ucrania sobre Protección de los Derechos del Consumidor (un producto de calidad adecuada, fabricado por encargo individual, no admite devolución ni cambio)." },
      ] },
      { h: "Archivos digitales (descarga de 3MF/STL)", blocks: [
        { p: "El importe de un archivo digital no se reembolsa una vez que el archivo ha sido generado y puesto a disposición para su descarga, ya que el servicio ya se ha prestado en su totalidad. Si el archivo no se puede descargar por motivos técnicos o está dañado por nuestra parte, lo regeneraremos gratuitamente o te devolveremos el importe." },
      ] },
      { h: "Impresión bajo pedido (producto físico)", blocks: [
        { ul: [
          "Antes del inicio de la producción: puedes cancelar el pedido y obtener un reembolso íntegro si aún no hemos empezado la impresión.",
          "Tras el inicio de la producción: el importe no se reembolsa, ya que el producto se fabrica de forma personalizada para tu pedido.",
          "Defecto, daño durante el transporte o disconformidad con el pedido: reimprimimos y enviamos el producto de nuevo gratuitamente, o te devolvemos el importe íntegro (según tu elección).",
        ] },
      ] },
      { h: "Cómo tramitar una devolución", blocks: [
        { p: "Escribe a {email} o llama al {phone}, indicando el número de pedido, el motivo y (en caso de defecto) una foto del producto. Examinaremos tu solicitud en un plazo de 1–3 días laborables." },
        { p: "El reembolso se realiza por el mismo medio con el que se efectuó el pago (reembolso a la tarjeta a través de LiqPay), dentro de los plazos previstos por las normas del sistema de pago y del banco." },
      ] },
      { h: "Contacto", blocks: [
        { p: "Este documento forma parte inseparable del [offer:Contrato de oferta pública]. Consultas: {email}." },
      ] },
    ],
  },

  delivery: {
    title: "Pago y entrega",
    sections: [
      { h: "Productos y precios", blocks: [
        { p: "Precios de los productos (el envío se cobra aparte según la tarifa del transportista):" },
        { ul: [
          "Llavero-mapa: desde 120 ₴ (≈ 3 €).",
          "Mapa 3D de barrio: S 5,5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴ (mapas: desde 6 €).",
          "Imán de nevera (mapa): 150 ₴.",
          "Descarga del archivo terminado 3MF/STL para imprimirlo por tu cuenta: gratuita dentro del límite de la cuenta.",
        ] },
      ] },
      { h: "Pago", blocks: [
        { p: "Pago en línea con tarjeta bancaria Visa / Mastercard a través del servicio seguro LiqPay. Los datos de la tarjeta se procesan en el lado del sistema de pago; nosotros no los almacenamos. También es posible pagar por acuerdo con el operador. El archivo digital dentro del límite gratuito se facilita sin coste." },
      ] },
      { h: "Fabricación", blocks: [
        { p: "Los productos se imprimen bajo pedido en bioplástico Eco PLA. Plazo orientativo de fabricación: 1–3 días laborables tras la confirmación del pedido (según la carga de trabajo y la complejidad)." },
      ] },
      { h: "Entrega", blocks: [
        { p: "Ucrania:" },
        { ul: ["Nova Poshta: sucursal o consigna automática.", "Ukrposhta: sucursal."] },
        { p: "Unión Europea (15 países):" },
        { ul: ["Nova Post EU.", "Meest."] },
        { p: "El coste de envío se calcula según las tarifas del transportista y se paga por separado (habitualmente al recibirlo). Plazo orientativo de entrega dentro de Ucrania: 2–4 días laborables tras el envío." },
      ] },
      { h: "Devoluciones", blocks: [
        { p: "Las condiciones de reembolso se describen en la página [refund:«Devolución y reembolso de fondos»]. Las condiciones generales, en el [offer:Contrato de oferta pública]." },
      ] },
      { h: "Contacto", blocks: [
        { p: "Consultas sobre el pago o la entrega: {email}, {phone}." },
      ] },
    ],
  },

  contacts: {
    title: "Contacto y datos fiscales",
    sections: [
      { h: "Cómo contactar con nosotros", blocks: [
        { kv: [
          { k: "Email", v: "{email}" },
          { k: "Teléfono", v: "{phone}" },
          { k: "Sitio web", v: "{domain}" },
        ] },
        { p: "Horario de procesamiento de pedidos: a diario, respondemos en un plazo de 24 horas." },
      ] },
      { h: "Tienda", blocks: [
        { kv: [
          { k: "Nombre", v: "{storeName}" },
          { k: "Dirección de la tienda", v: "{storeAddress}" },
        ] },
      ] },
      { h: "Vendedor (empresario individual)", blocks: [
        { kv: [
          { k: "Nombre", v: "{ownerFull}" },
          { k: "Número fiscal (IPN/RNOKPP)", v: "{taxId}" },
          { k: "Actividad (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
        ] },
      ] },
      { h: "Qué vendemos", blocks: [
        { p: "{storeName} ofrece modelos 3D de mapas de ciudades y llaveros-mapa bajo pedido. Puedes descargar el archivo terminado para imprimir (3MF/STL) dentro del límite gratuito o encargar la impresión de un producto de bioplástico Eco PLA con envío. Los precios orientativos se indican en el configurador y en la página [delivery:«Pago y entrega»]." },
      ] },
      { h: "Documentos", blocks: [
        { ul: [
          "[offer:Contrato de oferta pública]",
          "[refund:Devolución y reembolso de fondos]",
          "[delivery:Pago y entrega]",
          "[privacy:Política de privacidad]",
          "[terms:Condiciones de uso]",
        ] },
      ] },
    ],
  },

  privacy: {
    title: "Política de privacidad",
    sections: [
      { h: "Qué datos recopilamos", blocks: [
        { p: "Nombre, email o número de teléfono (al iniciar sesión o registrarte), así como los datos de contacto y la dirección de entrega que indicas al tramitar un pedido. Los datos técnicos (el historial de modelos generados) se guardan en tu cuenta." },
      ] },
      { h: "Cómo utilizamos los datos", blocks: [
        { p: "Exclusivamente para prestar el servicio: inicio de sesión en la cuenta, conservación del historial de modelos, tramitación y entrega de pedidos y contacto contigo respecto al pedido. No vendemos ni cedemos tus datos a terceros con fines publicitarios." },
      ] },
      { h: "Rutas cargadas (GPX) y geodatos", blocks: [
        { p: "Si cargas un archivo GPX (por ejemplo, la exportación de tu propia actividad desde Strava u otra aplicación), procesamos las coordenadas de la ruta exclusivamente para construir tu modelo 3D. Estos son tus propios datos: no los publicamos, no los cedemos a terceros ni los usamos con fines publicitarios. Los puntos de la ruta se simplifican y se conservan solo el tiempo necesario para la generación y (si has iniciado sesión) para mantener el historial de modelos en tu cuenta; puedes solicitar su eliminación en cualquier momento." },
        { p: "La búsqueda de un lugar en el mapa envía tu consulta al servicio de geocodificación de OpenStreetMap (Nominatim), y los propios mapas se cargan desde los tiles de OpenStreetMap, conforme a sus condiciones de uso. No facilitamos a estos servicios tu nombre ni tus datos de contacto." },
      ] },
      { h: "Almacenamiento y servicios", blocks: [
        { p: "La autenticación funciona mediante Google Firebase Authentication. El sitio está protegido por Cloudflare. Los pedidos se procesan manualmente. Los datos se guardan en un servidor seguro solo el tiempo necesario para ejecutar el pedido y mantener el historial." },
      ] },
      { h: "Cookies y analítica", blocks: [
        { p: "Utilizamos la analítica privada de Cloudflare, sin cookies publicitarias de terceros. Las cookies se emplean únicamente para el funcionamiento del inicio de sesión en la cuenta." },
      ] },
      { h: "Tus derechos", blocks: [
        { p: "Puedes solicitar la eliminación de tu cuenta y de los datos asociados. Escríbenos a {email}." },
      ] },
      { h: "Contacto", blocks: [
        { p: "Para cuestiones de privacidad: {email}." },
      ] },
    ],
  },

  terms: {
    title: "Condiciones de uso",
    sections: [
      { h: "Sobre el servicio", blocks: [
        { p: "{storeName} te permite crear un modelo 3D de una zona de la ciudad o de un llavero-mapa a partir de los datos abiertos de OpenStreetMap y descargar el archivo terminado para impresión 3D (3MF/STL) o encargar la impresión." },
      ] },
      { h: "Cuenta y descargas gratuitas", blocks: [
        { p: "Para descargar el modelo completo se necesita una cuenta. Cada usuario dispone de 5 descargas gratuitas. A partir de ahí, por acuerdo (impresión/pago), con contacto a través del sitio." },
      ] },
      { h: "Datos y derechos de autor", blocks: [
        { p: "Datos cartográficos © OpenStreetMap contributors (ODbL). Los archivos generados puedes utilizarlos para impresión personal. La reventa del servicio o el uso comercial masivo requieren un acuerdo aparte." },
      ] },
      { h: "Pedidos y pago", blocks: [
        { p: "El pedido se tramita a través del sitio; el pago, en línea mediante LiqPay o por acuerdo. Los detalles de precios y entrega están en la página [delivery:«Pago y entrega»]; las condiciones completas, en el [offer:Contrato de oferta pública]." },
      ] },
      { h: "Responsabilidad", blocks: [
        { p: "El servicio se presta «tal cual». Procuramos la máxima precisión de los modelos, pero no garantizamos su correspondencia total con los objetos reales debido a las limitaciones de los datos de origen de OSM." },
      ] },
      { h: "Contacto", blocks: [
        { p: "Consultas: {email}." },
      ] },
    ],
  },
};
