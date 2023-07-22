var blogViewingLink = null;
var blogCtr = 0;
var bgColors = [
				// '#f44336',
				'#e91e63', '#9c27b0', 
				// '#ff1744',
				// '#ff4081', 
				// '#aa00ff', 
				// '#3f51b5', 
				'#673ab7', 
				// '#2196f3', 
				// '#651fff', 
				// '#b388ff',
				// '#304ffe', 
				// '#03a9f4', 
				'#00838f', 
				// '#009688',
				'#00b0ff', 
				// '#18ffff', 
				// '#64ffda', 
				// '#1de9b6', 
				//'#eeff41',
				// '#ef6c00','#ff3d00'
				];

var months = [
"Jan", "Feb", "Mar",
"Apr", "May", "Jun", "Jul",
"Aug", "Sep", "Oct",
"Nov", "Dec"
];

function hexToRgb(hex) {
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, function(m, r, g, b) {
        return r + r + g + g + b + b;
    });

    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function rgbString(r,g,b,a=1.0){
	return "rgba("+r+","+g+","+b+","+a+")";
}

function hexToRgbFn(hex, a = 0){
	var res = hexToRgb(hex);
	var r = 0, g=0, b=0;
	if(hex!=null){
		r = res.r;
		g=res.g;
		b=res.b;
	}
	return rgbString(r,g,b,a);
}

$().ready(function(){
	
	var converter = new showdown.Converter();
	
	// loadColors();
	$('#blogsContainer').ready(function(){
		// alert("hi");
		// loadMoreBlogs();
	});
	// hljs.initHighlightingOnLoad();

	// replaceAllDates();
});

// For debugginh
function loadColors(){
	shuffleArray(bgColors);
	$('#loading').css('display','none');
	var blogsContainer = $('#blogsContainer')[0];
	for(var i=0;i<bgColors.length;i++){
		var blog = {slug:'abc',title: bgColors[i], updated_at:'2017-08-27T14:00:00Z'};
		var b = createBlogCard(blog);
		blogsContainer.appendChild(b);	
		$('#'+b.id).animate({opacity:1},100);	
	}
}

function replaceAllDates(){
	var dates = $('.catchDate');
	if(dates.length==0){
		return;
	}
	for(var dt of dates){
		dt.innerText = convertDateFromPHPToMy(dt.innerText);
	}
}

// inclusive min, exclusive max
function getRandomInt(minnum,maxnum){
	var mi = Math.floor(minnum);
	var ma = Math.ceil(maxnum);
	return Math.floor(Math.random() * (ma-mi)) + mi;
}

function shuffleArray(inpArr){
	for(var i=0;i<inpArr.length-1; i++){
		var exchgInd = getRandomInt(i,inpArr.length);	
		var tmp = inpArr[i];
		inpArr[i]=inpArr[exchgInd];
		inpArr[exchgInd] = tmp;
	}
}

function createBlogLinkFromSlug(slug){
	if(blogViewingLink==null){
		blogViewingLink = $('#blogViewingLink').data('link');
	}
	return blogViewingLink + "/" + slug;
}


function createUTCTime(timestr){
	var dt = timestr.split(" ");
	return dt[0] + "T" + dt[1] + "Z";
}

function formatDate(date){
	return months[date.getMonth()] + " " + date.getDate() + ", " + date.getFullYear();
}

function convertDateFromPHPToMy(timestr){
	var dstr = createUTCTime(timestr);
	var dt = new Date(dstr);
	return formatDate(dt);
}

function createBlogCard(blog){
	// the link to the blog
	var bloglink = document.createElement('a');
	bloglink.href = createBlogLinkFromSlug(blog.slug);
	// date
	var dateElem = document.createElement('div');
	// var justDate = new Date(createUTCTime(blog.updated_at));

	dateElem.innerText = convertDateFromPHPToMy(blog.updated_at);
	dateElem.classList.add("date");
	dateElem.classList.add("lead");
	// the main blog card container
	var blogElem = document.createElement('div');
	var wrapper = document.createElement('div');
	blogElem.id = blogCtr+"";
	// console.log(bgColors[0]);
	blogElem.style.background =  hexToRgbFn(bgColors[blogCtr%bgColors.length], 1.0);
	blogCtr++;
	blogElem.className = 'blogCard';
	blogElem.classList.add('jumbotron');
	// title
	var title = document.createElement('h1');
	title.classList.add("display-3");
	bloglink.innerText = blog.title;
	title.appendChild(bloglink);
	// blogElem.appendChild(title);
	// blogElem.appendChild(dateElem);
	wrapper.className = "content-wrapper";
	wrapper.appendChild(title);
	wrapper.appendChild(dateElem);
	blogElem.appendChild(wrapper);
	blogElem.style.opacity=0;
	//TODO render date
	return blogElem;
}


var fontToggle = true;

function switchFonts(){
	var ff = fontToggle? "'Open Sans', sans-serif": "'Roboto', sans-serif";
	$('#body').css('font-family',ff);
	fontToggle = !fontToggle;
}