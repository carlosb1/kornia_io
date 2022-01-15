use kornia_io::cv;

fn main() {
    println!("Hello, world!");
    let shape: Vec<usize> = vec![1, 1, 2, 2];
    let data: Vec<u8> = (0..cv::cumprod(&shape)).map(|x| x as u8).collect();
    let t1 = cv::Tensor::new(shape, data);
    println!("{:?}", t1);
    println!("The tensor has {} dimensions.", t1.dims());

    // loop tensor
    println!("Print tensor");
    t1.print();

    // clone a tensor
    let t2 = t1.clone();

    println!("Sum");
    let t3 = t1.add(t2.clone());
    t3.print();
    println!("Mul");
    let t4 = t1.mul(t2.clone());
    t4.print();

    println!("Subs");
    let t5 = t1.subs(t2.clone());
    t5.print();

    println!("Div");
    let t6 = t1.div(t2.clone());
    t6.print();

    // load image from file
    let img = cv::Tensor::from_file("/home/edgar/Downloads/g279.png");
    println!("{:?}", img);

    println!("Goodbye, world!");
}
