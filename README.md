```
// ------------------ Declaring variables --------------------
// Mutable variables:
let x <int> = 1;
bez z <float> = 1.1; // bez is the same as let

// Immutable variables:

const x <int> = 1; // Cannot be changed
beton y <float> = 1.1; // Cannot be changed

// ------------------ Functions ------------------
// Function calls:
printf("Hello world");
let val <auto> = add(10, 20);

// Function definition:
fun main() <noret> { // noret is a keyword representing `no return` logic

}

fun print_2_values(x: <int>, y: <float>) <noret> {

  printf("%d",x + std_conv<int>(y));
}

// Function with return and arguments:

fun add(x: <int>, y: <int>) <int> {
  return x + y;
}


// Macro call:
$plus(10,20);
// Macro definition: 
@macro plus(x, y) {
  return std_conv<int>(x) + std_conv<int>(y);
}

// --------------- loops ---------------

// while loop:

while (1 == 1){

break; // other keywords: break, continue

}

// for loops:

for (let i <int> = 0; i < 10; i++){

printf("%d",i);

}

// for each loops: (WIP)


// ----------------- if else ----------------

// if:

if (i == 10){

  printf("i is 10");

}

// if else:

if (i == 10){
  printf("i is 10");
} else {
printf("i is not 10");
}

// if else if

if (i == 10){

printf("i is 10");

} elseif (i == 11){

printf("i is 11");

} else {

printf("i is not 10");

}

// --------------- Data Structures ----------------

// structs: Structs are default data structures and there can me struct methods too OOP

// Simple struct:

struct Point {
  x <float>,
  y <float>
}

// Struct usage:
let p <Point> = Point { x:1.0, y:0.0};


// Struct element access:
printf("%f",p.x);

// Struct OOP:
struct Point{
  x <int>,
  y <int>,
  fun get_x(self: <Point>) <int> {
    return self.x;
}
  fun copy(self: <Point>) <Point> {
    let temp = Point {x:self.x, y:self.y};
    return temp;
}
}

// ----------------- Enums ------------------

enum Status {
Ok, NotOk, ProbablyOk
}


// In this programming language there is no such thing as null value all not assigned declarations are assigned to default values (even structured data and special values)

// Some more advanced examples:
// C like macros
 #cdef Sum(x, y) ((x) + (y))

struct MyStruct{
}

fun main() <noret> {
	let x <auto> = MyStruct{};
	let y <MyStruct> = MyStruct{};
	let b <auto> = 10;
	if (typeof(y) == typeof(x)){
		printf("True");
	}
	printf("%d", typeof(b));
}
// Generics
fun add<T, U>(x: <T>, y: <U>) <T> {
	return x + std_conv<int>(y);
}

fun main() <noret> {
	printf("%d", add(10,20.0));
}
// Heap allocation

fun main() <noret> {
    let heap_int_ptr <&int>;

    heap_int_ptr = new <int>;

    *heap_int_ptr = 42;

    printf("Value from heap: %d\n", *heap_int_ptr);

    let another_heap_ptr <&int> = new <int>;
    *another_heap_ptr = 123;
    printf("Another heap value: %d\n", *another_heap_ptr);

    delete heap_int_ptr;
    delete another_heap_ptr;
}
// C libraries import
import_c "stdio";

extern fun getchar() <int>;

fun main() <noret> {
	let c <int> = getchar();
	printf("%c", c);
}
// Imports
import "test_enum.popo" as mymod;

fun main() <noret> {
	let x <mymod::Status> = mymod::Status::Ok;
	let y <mymod::Day> = mymod::Day{day_name: "Hello"};
	mymod::check_status(x);
	y.show_day();
}
// Arrays and pointers
fun main() <noret> {
    let numbers <[int]> = [10, 20, 30, 40, 50];
    let myarr <[int, 32]>;
    let arr2 <[int, 5]> = [1,2,3,4,5];

    let first <auto> = numbers[0];
    let third <auto> = numbers[2];
    numbers[1] = 25; 

    let p_int <&int>;
    let pp_int <&(&int)>; 

    let ptr_array <[&int]> = [&numbers[0], &numbers[1]];
    let first_ptr_val <auto> = *ptr_array[0];

    let x <int> = 100;
    p_int = &x;
    let val_from_ptr <auto> = *p_int;
    *p_int = 200; 


    let matrix <[[int,2], 2]> = [[1,2], [3,4]];
    let m_val <auto> = matrix[0][1]; 

    struct Data {
        items <[int,3]>,
        config_ptr <&int>
    }

    let d_ <auto> = Data{ items: [1,2,3], config_ptr: &x };
    let d_item <auto> = d_.items[0];
    let d_config <auto> = *d_.config_ptr;

    let auto_arr <auto> = [7.0, 8.0, 9.0]; 
    let auto_ptr <auto> = &first;         
    let auto_idx <auto> = auto_arr[1];   
}
// sizeof
fun main() <noret>{
	let x <int>;
	let y <int> = sizeof(x);
	printf("x sizeof = %d\n", y);
	printf("float sizeof = %d", sizeof(<float>));
}
// Type annotation

fun main() <noret> {
	let x <int(128)> = 10000000;

	printf("%d", x);
}
// type conversion

fun main() <noret> {
	let x <float> = 10.2;
	let y <int> = 5;
	let z <auto> = std_conv<float>(y) + x;
	printf("%f", z);
}

```
