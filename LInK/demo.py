
draw_html = '''
<html>
<head>
  <style>
    footer {visibility: hidden}
    .svg_pad {
      display: block;
      height: 350px;
      width: 350px;
      background: #ebebeb;
      border: 1px solid #9b9b9b;
      border-radius: 0.5em;
    }
    #sketch {
      background: #fff;
      border-radius: 0.5em;
    }
    
  </style>

</head>
<body>
<h2>Draw Your Target Curve</h2>
<div class="svg_pad" id="svg_pad">
  <svg  viewBox="0 0 350 350"  xmlns="http://www.w3.org/2000/svg" id="sketch">

  </svg>
</div>

</body>

</html>
'''

draw_script = '''

function test(){
    let script1 = document.createElement('script');
    script1.src = "https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js";
    document.head.appendChild(script1); 
    let script = document.createElement('script');
    
    script.innerHTML = `
        var lastEvent;
        var mouseDown = false;
        var path = [];
        var path_start;
        var keys = {37: 1, 38: 1, 39: 1, 40: 1};

        function downloadObjectAsJson(exportName){
            var JStr = document.getElementById("json_text").querySelectorAll("textarea")[0].value;
            console.log(JStr);
            var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(JSON.parse(JStr)));
            var downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href",     dataStr);
            downloadAnchorNode.setAttribute("download", exportName + ".json");
            document.body.appendChild(downloadAnchorNode); // required for firefox
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }
        
        function preventDefault(e) {
          e.preventDefault();
        }

        function preventDefaultForScrollKeys(e) {
          if (keys[e.keyCode]) {
            preventDefault(e);
            return false;
          }
        }

        // modern Chrome requires { passive: false } when adding event
        var supportsPassive = false;
        try {
          window.addEventListener("test", null, Object.defineProperty({}, 'passive', {
            get: function () { supportsPassive = true; } 
          }));
        } catch(e) {}

        var wheelOpt = supportsPassive ? { passive: false } : false;
        var wheelEvent = 'onwheel' in document.createElement('div') ? 'wheel' : 'mousewheel';

        // call this to Disable
        function disableScroll() {
          window.addEventListener('DOMMouseScroll', preventDefault, false); // older FF
          window.addEventListener(wheelEvent, preventDefault, wheelOpt); // modern desktop
          window.addEventListener('touchmove', preventDefault, wheelOpt); // mobile
          window.addEventListener('keydown', preventDefaultForScrollKeys, false);
        }

        // call this to Enable
        function enableScroll() {
          window.removeEventListener('DOMMouseScroll', preventDefault, false);
          window.removeEventListener(wheelEvent, preventDefault, wheelOpt); 
          window.removeEventListener('touchmove', preventDefault, wheelOpt);
          window.removeEventListener('keydown', preventDefaultForScrollKeys, false);
        }
        
        document.getElementById("sketch").addEventListener("touchstart", (e) => {
          document.getElementById("sketch").innerHTML = "";
          var bcr = e.target.getBoundingClientRect();
          disableScroll();
          path = [];
          lastEvent = e;
          path_start = e;
          mouseDown = true;
          console.log(lastEvent);
          path.push([lastEvent.touches[0].clientX,lastEvent.touches[0].clientY]);
        });

        document.getElementById("sketch").addEventListener("touchmove", (e) => {
          var bcr = e.target.getBoundingClientRect();
          if(mouseDown){
            document.getElementById("sketch").innerHTML += '<line x1="' + (lastEvent.touches[0].clientX - bcr.x) + '" y1="' + (lastEvent.touches[0].clientY - bcr.y) + '" x2="' + (e.touches[0].clientX - bcr.x) + '" y2="' + (e.touches[0].clientY - bcr.y) + '" stroke="black" />';
            lastEvent = e;
            path.push([lastEvent.touches[0].clientX,lastEvent.touches[0].clientY]);
          }
        });

        document.getElementById("sketch").addEventListener("touchend", (e) => {
          var bcr = e.target.getBoundingClientRect();
          if(!$('#partial input')[0].checked){
            document.getElementById("sketch").innerHTML += '<line x1="' + (lastEvent.touches[0].clientX - bcr.x) + '" y1="' + (lastEvent.touches[0].clientY - bcr.y) + '" x2="' + (path_start.touches[0].clientX - bcr.x) + '" y2="' + (path_start.touches[0].clientY - bcr.y) + '" stroke="black" />';
            path.push([path_start.touches[0].clientX,path_start.touches[0].clientY]);
          }
          lastEvent = e;
          mouseDown = false;
          enableScroll();
        });

        document.getElementById("sketch").addEventListener("mousedown", (e) => {
          document.getElementById("sketch").innerHTML = ""
          path = [];
          lastEvent = e;
          path_start = e;
          mouseDown = true;
          console.log(lastEvent);
          path.push([lastEvent.offsetX,lastEvent.offsetY]);
        });

        document.getElementById("sketch").addEventListener("mousemove", (e) => {
          if(mouseDown){
            document.getElementById("sketch").innerHTML += '<line x1="' + lastEvent.offsetX + '" y1="' + lastEvent.offsetY + '" x2="' + e.offsetX + '" y2="' + e.offsetY + '" stroke="black" />';
            lastEvent = e;
            path.push([lastEvent.offsetX,lastEvent.offsetY]);
          }
        });
        
        document.getElementById("sketch").addEventListener("mouseup", (e) => {
          if(!$('#partial input')[0].checked){
            document.getElementById("sketch").innerHTML += '<line x1="' + lastEvent.offsetX + '" y1="' + lastEvent.offsetY + '" x2="' + path_start.offsetX + '" y2="' + path_start.offsetY + '" stroke="black" />';
            path.push([path_start.offsetX,path_start.offsetY]);
          }
          lastEvent = e;
          mouseDown = false;
          $("#curve_out textarea")[0].value = path.toString();
        });`;
    document.head.appendChild(script);
}
'''

css = """
div:has(>.clr_btn) {max-width: 350px !important}
# .prog .output-class{display:none !important;}
# .prog .confidence-set{margin-top:10px;}
.plotpad{padding-top: 15px;}

.intro p{
  font-size: 1.2em;
}
"""