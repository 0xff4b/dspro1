export default function ({parentElement, data, setStateValue}) {
  const input=parentElement.querySelector('#address-input');
  const list=parentElement.querySelector('#address-results');
  const status=parentElement.querySelector('#address-status');
  if (!input.dataset.initialized) {
    input.value=data?.initial || '';
    input.dataset.confirmed=data?.initial || '';
    input.dataset.initialized='true';
  }
  let timer, controller, version=0, items=[], active=-1;
  const close=()=>{
    list.hidden=true;
    input.setAttribute('aria-expanded','false');
    input.removeAttribute('aria-activedescendant');
  };
  const highlight=(index)=>{
    active=index;
    [...list.children].forEach((node,i)=>node.setAttribute('aria-selected',String(i===active)));
    if(active>=0) {
      input.setAttribute('aria-activedescendant',list.children[active].id);
      list.children[active].scrollIntoView({block:'nearest'});
    }
  };
  const choose=(index)=>{
    const item=items[index]; if(!item || input.dataset.confirmed===item) return;
    ++version; clearTimeout(timer); controller?.abort();
    input.value=item; input.dataset.confirmed=item;
    close(); status.textContent='Adresse bestätigt. Gebäude- und Wohnungsdaten werden geladen.';
    setStateValue('selected',item);
  };
  const search=async(query,requestVersion)=>{
    const requestController=new AbortController(); controller=requestController;
    const timeout=setTimeout(()=>requestController.abort(),10000);
    status.textContent='Adressen werden gesucht …';
    try {
      const url=new URL('https://api3.geo.admin.ch/rest/services/ech/SearchServer');
      url.search=new URLSearchParams({searchText:query,type:'locations',origins:'address',sr:'2056',limit:'8'});
      const response=await fetch(url,{signal:requestController.signal,credentials:'omit'});
      if(!response.ok) throw new Error('Address service unavailable');
      const payload=await response.json();
      if(requestVersion!==version) return;
      // External labels are text, never executable HTML.
      items=[...new Set((payload.results||[]).map(r=>String(r.attrs?.label||'').replace(/<[^>]*>/g,'').trim()).filter(Boolean))];
      list.replaceChildren(); active=-1;
      items.forEach((label,index)=>{
        const option=document.createElement('li');
        option.id='address-option-'+index; option.setAttribute('role','option');
        option.setAttribute('aria-selected','false'); option.textContent=label;
        let pointerStart=null;
        option.addEventListener('pointerdown',event=>{
          pointerStart={x:event.clientX,y:event.clientY,id:event.pointerId};
          event.preventDefault(); // Keep focus in the combobox.
        });
        option.addEventListener('pointercancel',()=>{pointerStart=null;});
        option.addEventListener('pointerup',event=>{
          if(pointerStart && pointerStart.id===event.pointerId &&
             Math.hypot(event.clientX-pointerStart.x,event.clientY-pointerStart.y)<10) {
            event.preventDefault(); choose(index);
          }
          pointerStart=null;
        });
        option.addEventListener('click',()=>choose(index));
        list.appendChild(option);
      });
      list.hidden=!items.length || parentElement.activeElement!==input; input.setAttribute('aria-expanded',String(!list.hidden));
      status.textContent=items.length ? 'Bitte die vollständige Adresse aus der Liste bestätigen.' : 'Keine Adresse gefunden. Bitte Ort oder Postleitzahl ergänzen.';
    } catch(error) {
      if(requestVersion!==version) return;
      close(); status.textContent='Adressdienst momentan nicht erreichbar. Bitte erneut tippen oder die Beispielwohnung nutzen.';
    } finally {clearTimeout(timeout);}
  };
  input.oninput=()=>{
    ++version; clearTimeout(timer); controller?.abort(); close(); items=[];
    if(input.dataset.confirmed) {
      input.dataset.confirmed='';
      setStateValue('selected',null);
    }
    const query=input.value.trim();
    if(query.length<3) {status.textContent='Bitte mindestens drei Zeichen eingeben.'; return;}
    const requestVersion=version;
    timer=setTimeout(()=>search(query,requestVersion),350);
  };
  input.onkeydown=(event)=>{
    if(event.key==='Escape') {close(); return;}
    if(list.hidden && items.length && (event.key==='ArrowDown' || event.key==='ArrowUp')) {list.hidden=false; input.setAttribute('aria-expanded','true');}
    if(!list.hidden && items.length) {
      if(event.key==='ArrowDown') {event.preventDefault(); highlight((active+1)%items.length);}
      if(event.key==='ArrowUp') {event.preventDefault(); highlight((active-1+items.length)%items.length);}
      if(event.key==='Enter') {event.preventDefault(); if(active>=0) choose(active); else status.textContent='Bitte einen Vorschlag anklicken oder mit Pfeiltasten und Enter bestätigen.';}
    }
  };
  input.onblur=()=>{close();};
  input.onfocus=()=>{if(items.length && !input.dataset.confirmed) {list.hidden=false;input.setAttribute('aria-expanded','true');}};
  if(input.dataset.confirmed) status.textContent='Bestätigt: '+input.dataset.confirmed;
  else if(input.value.trim().length>=3) { const current=version; timer=setTimeout(()=>search(input.value.trim(),current),350); }
  return ()=>{++version;clearTimeout(timer);controller?.abort();input.oninput=null;input.onkeydown=null;input.onblur=null;input.onfocus=null;};
}