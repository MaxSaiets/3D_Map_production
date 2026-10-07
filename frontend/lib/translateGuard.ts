/**
 * 07.10.2026: автопереклад Chrome (Google Translate) валив конструктори.
 *
 * Перекладач замінює текстові вузли сторінки своїми `<font>`-обгортками. React
 * про це не знає і при наступній зміні розмітки (клік «Згенерувати» → кнопка
 * перетворюється на прогрес) кличе `insertBefore`/`removeChild` з вузлом, якого
 * вже немає на місці → `NotFoundError` → error boundary «Щось пішло не так».
 * Реальний випадок: відвідувач із Молдови, сторінка /en/keychains, Chrome
 * перекладав на російську — цілий день жодна генерація не доходила до кінця.
 *
 * Стандартний обхід (facebook/react#11538): робимо ці дві операції стійкими до
 * чужих мутацій DOM. Скрипт інлайниться в <head> і виконується ДО гідрації.
 *  - removeChild: вузол уже не дитина цього батька → прибираємо його звідти, де
 *    він є (або нічого, якщо він від'єднаний), без винятку.
 *  - insertBefore: опорний вузол переїхав усередину обгортки перекладача →
 *    вставляємо перед тією дитиною цього батька, що його містить; якщо опорний
 *    вузол зовсім від'єднаний — дописуємо в кінець (видно, хай і не на місці).
 */
export const TRANSLATE_GUARD_SCRIPT = `(function(){try{
if(typeof Node!=="function"||!Node.prototype||Node.prototype.__mdGuard)return;
var P=Node.prototype,rc=P.removeChild,ib=P.insertBefore;P.__mdGuard=1;
P.removeChild=function(c){if(c&&c.parentNode!==this){if(c.parentNode)return rc.call(c.parentNode,c);return c}return rc.apply(this,arguments)};
P.insertBefore=function(n,r){if(r&&r.parentNode!==this){var a=r.parentNode;while(a&&a.parentNode!==this)a=a.parentNode;return ib.call(this,n,a||null)}return ib.apply(this,arguments)};
}catch(e){}})();`;
