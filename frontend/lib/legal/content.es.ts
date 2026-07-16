import type { LegalSet } from "./content";

export const es: LegalSet = {
  offer: {
    title: "Contrato de oferta pública",
    intro: [
      "Este documento constituye una propuesta pública oficial (oferta) de {ownerFull} (en adelante, el «Vendedor») para celebrar un contrato de compraventa de productos y de prestación de servicios en las condiciones que se exponen a continuación. Al pagar un pedido en el sitio {domain} (en adelante, el «Sitio»), el Comprador confirma que ha leído íntegramente, ha comprendido y acepta sin reservas las condiciones de este Contrato (aceptación de la oferta conforme a los art. 633, 641 y 642 del Código Civil de Ucrania).",
    ],
    sections: [
      { h: "1. Definiciones", blocks: [
        { p: "Vendedor: {ownerShort}, número fiscal {taxId}." },
        { p: "Comprador: cualquier persona con capacidad jurídica que haya realizado un pedido en el Sitio." },
        { p: "Producto / Servicio: el modelo 3D digital (archivo 3MF/STL) de un mapa de ciudad o de un llavero-mapa, y/o la fabricación (impresión 3D) de un producto físico de bioplástico Eco PLA por encargo individual del Comprador." },
      ] },
      { h: "2. Objeto del contrato", blocks: [
        { p: "El Vendedor se compromete a facilitar al Comprador el modelo 3D digital y/o a fabricar y entregar el producto físico bajo pedido, y el Comprador, a recibirlos y pagarlos conforme a las condiciones de este Contrato." },
        { p: "Cada producto se fabrica de forma individual según los parámetros (lugar en el mapa, forma, tamaño, texto) que el Comprador elige por sí mismo en el configurador del Sitio, por lo que constituye un producto fabricado por encargo." },
      ] },
      { h: "3. Tramitación del pedido", blocks: [
        { p: "El Comprador crea el pedido en el configurador del Sitio e indica sus datos de contacto (nombre, teléfono, modo y dirección de entrega). El pedido se considera aceptado tras su pago o tras la confirmación por parte de un operador." },
        { p: "El Comprador es responsable de la veracidad de los datos facilitados. El Vendedor no se hace responsable de las consecuencias derivadas de errores en los datos proporcionados por el Comprador." },
      ] },
      { h: "4. Precio y pago", blocks: [
        { p: "Los precios de los productos y servicios se indican en el Sitio en grivnas (para pedidos dentro de Ucrania) y en euros (para envíos a la UE) y son orientativos hasta la confirmación del pedido. El Comprador ve el importe definitivo en el paso de tramitación." },
        { p: "El pago se realiza en línea a través del servicio de pago LiqPay (con tarjeta bancaria Visa/Mastercard) o por otro medio acordado. La descarga del archivo digital terminado dentro del límite gratuito es gratuita." },
        { p: "Más detalles en la página [delivery:«Pago y entrega»]." },
      ] },
      { h: "5. Fabricación y entrega", blocks: [
        { p: "El archivo digital se facilita en la cuenta o por correo electrónico de inmediato o tras la confirmación del pedido. El producto físico se fabrica y se envía en el plazo indicado en la página «Pago y entrega», mediante los servicios Nova Poshta o Ukrposhta (Ucrania), Nova Post EU o Meest (UE)." },
      ] },
      { h: "6. Devolución de fondos", blocks: [
        { p: "Dado que los productos se fabrican por encargo individual y que los archivos digitales tienen carácter de contenido electrónico, las devoluciones se rigen por un documento aparte, [refund:«Devolución y reembolso de fondos»], que forma parte inseparable de este Contrato." },
      ] },
      { h: "7. Derechos de propiedad intelectual", blocks: [
        { p: "Los datos cartográficos proceden de OpenStreetMap (ODbL); los datos de altitud, de fuentes abiertas. El modelo 3D generado se facilita al Comprador para uso e impresión personales y no comerciales. La ruta GPX cargada por el Comprador son sus propios datos y se procesan exclusivamente para construir el modelo (véase la [privacy:Política de privacidad])." },
      ] },
      { h: "8. Responsabilidad de las partes", blocks: [
        { p: "El Vendedor no responde de la calidad de impresión en el equipo del Comprador cuando este imprime por su cuenta el archivo descargado. La responsabilidad total del Vendedor se limita al importe del pedido pagado." },
        { p: "Las partes quedan exoneradas de responsabilidad por el incumplimiento de sus obligaciones a causa de circunstancias de fuerza mayor." },
      ] },
      { h: "9. Datos personales", blocks: [
        { p: "Al realizar el pedido, el Comprador da su consentimiento para el tratamiento de sus datos personales con el fin de ejecutar el pedido, conforme a la Ley de Ucrania sobre Protección de Datos Personales y a la [privacy:Política de privacidad]." },
      ] },
      { h: "10. Resolución de conflictos y vigencia", blocks: [
        { p: "Los conflictos se resuelven mediante negociación y, de no alcanzarse un acuerdo, conforme a la legislación vigente de Ucrania. El Contrato está en vigor desde el momento de la aceptación y hasta el cumplimiento íntegro de las obligaciones por ambas partes. El Vendedor tiene derecho a modificar las condiciones publicando una nueva versión en esta página." },
      ] },
      { h: "11. Datos del Vendedor", blocks: [
        { kv: [
          { k: "Nombre", v: "{ownerFull}" },
          { k: "Número fiscal (IPN/RNOKPP)", v: "{taxId}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Dirección de registro", v: "{ownerRegAddress}" },
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
          { k: "Dirección de registro", v: "{ownerRegAddress}" },
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
